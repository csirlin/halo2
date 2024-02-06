//! Tools for developing circuits.

use std::cmp::max;
use std::cmp::min;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fmt;
use std::hash::Hasher;
use std::iter;
use std::ops::{Add, Mul, Neg, Range};
use ff::Field;
use std::hash::Hash;

use crate::plonk::AdviceQuery;
use crate::plonk::Assigned;
use crate::plonk::FixedQuery;
use crate::plonk::InstanceQuery;
use crate::plonk::VirtualCell;
use crate::poly::Rotation;
use crate::{
    circuit,
    plonk::{
        permutation, Advice, Any, Assignment, Circuit, Column, ConstraintSystem, Error, Expression,
        Fixed, FloorPlanner, Instance, Selector, circuit::Gate
    },
};

pub mod metadata;
mod util;

mod failure;
pub use failure::{FailureLocation, VerifyFailure};

pub mod cost;
pub use cost::CircuitCost;

mod gates;
pub use gates::CircuitGates;

mod tfp;
pub use tfp::TracingFloorPlanner;

#[cfg(feature = "dev-graph")]
mod graph;

#[cfg(feature = "dev-graph")]
#[cfg_attr(docsrs, doc(cfg(feature = "dev-graph")))]
pub use graph::{circuit_dot_graph, layout::CircuitLayout};

///
#[derive(Debug)]
pub struct Region {
    /// The name of the region. Not required to be unique.
    name: String,
    /// The columns involved in this region.
    columns: HashSet<Column<Any>>,
    /// The rows that this region starts and ends on, if known.
    rows: Option<(usize, usize)>,
    /// The selectors that have been enabled in this region. All other selectors are by
    /// construction not enabled.
    enabled_selectors: HashMap<Selector, Vec<usize>>,
    /// The cells assigned in this region. We store this as a `Vec` so that if any cells
    /// are double-assigned, they will be visibly darker.
    cells: Vec<(Column<Any>, usize)>,
}

impl Region {
    fn update_extent(&mut self, column: Column<Any>, row: usize) {
        self.columns.insert(column);

        // The region start is the earliest row assigned to.
        // The region end is the latest row assigned to.
        let (mut start, mut end) = self.rows.unwrap_or((row, row));
        if row < start {
            // The first row assigned was not at start 0 within the region.
            start = row;
        }
        if row > end {
            end = row;
        }
        self.rows = Some((start, end));
    }
}

/// The value of a particular cell within the circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellValue<F: Field> {
    /// An unassigned cell.
    Unassigned,
    /// A cell that has been assigned a value.
    Assigned(F),
    /// A unique poisoned cell.
    Poison(usize),
}

/// A value within an expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
enum Value<F: Field> {
    Real(F),
    Poison,
}

impl<F: Field> From<CellValue<F>> for Value<F> {
    fn from(value: CellValue<F>) -> Self {
        match value {
            // Cells that haven't been explicitly assigned to, default to zero.
            CellValue::Unassigned => Value::Real(F::ZERO),
            CellValue::Assigned(v) => Value::Real(v),
            CellValue::Poison(_) => Value::Poison,
        }
    }
}

impl<F: Field> Neg for Value<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Value::Real(a) => Value::Real(-a),
            _ => Value::Poison,
        }
    }
}

impl<F: Field> Add for Value<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Value::Real(a + b),
            _ => Value::Poison,
        }
    }
}

impl<F: Field> Mul for Value<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Value::Real(a * b),
            // If poison is multiplied by zero, then we treat the poison as unconstrained
            // and we don't propagate it.
            (Value::Real(x), Value::Poison) | (Value::Poison, Value::Real(x))
                if x.is_zero_vartime() =>
            {
                Value::Real(F::ZERO)
            }
            _ => Value::Poison,
        }
    }
}

impl<F: Field> Mul<F> for Value<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        match self {
            Value::Real(lhs) => Value::Real(lhs * rhs),
            // If poison is multiplied by zero, then we treat the poison as unconstrained
            // and we don't propagate it.
            Value::Poison if rhs.is_zero_vartime() => Value::Real(F::ZERO),
            _ => Value::Poison,
        }
    }
}

/// A test prover for debugging circuits.
///
/// The normal proving process, when applied to a buggy circuit implementation, might
/// return proofs that do not validate when they should, but it can't indicate anything
/// other than "something is invalid". `MockProver` can be used to figure out _why_ these
/// are invalid: it stores all the private inputs along with the circuit internals, and
/// then checks every constraint manually.
///
/// # Examples
///
/// ```
/// use group::ff::PrimeField;
/// use halo2_proofs::{
///     circuit::{Layouter, SimpleFloorPlanner, Value},
///     dev::{FailureLocation, MockProver, VerifyFailure},
///     pasta::Fp,
///     plonk::{Advice, Any, Circuit, Column, ConstraintSystem, Error, Selector},
///     poly::Rotation,
/// };
/// const K: u32 = 5;
///
/// #[derive(Copy, Clone)]
/// struct MyConfig {
///     a: Column<Advice>,
///     b: Column<Advice>,
///     c: Column<Advice>,
///     s: Selector,
/// }
///
/// #[derive(Clone, Default)]
/// struct MyCircuit {
///     a: Value<u64>,
///     b: Value<u64>,
/// }
///
/// impl<F: PrimeField> Circuit<F> for MyCircuit {
///     type Config = MyConfig;
///     type FloorPlanner = SimpleFloorPlanner;
///
///     fn without_witnesses(&self) -> Self {
///         Self::default()
///     }
///
///     fn configure(meta: &mut ConstraintSystem<F>) -> MyConfig {
///         let a = meta.advice_column();
///         let b = meta.advice_column();
///         let c = meta.advice_column();
///         let s = meta.selector();
///
///         meta.create_gate("R1CS constraint", |meta| {
///             let a = meta.query_advice(a, Rotation::cur());
///             let b = meta.query_advice(b, Rotation::cur());
///             let c = meta.query_advice(c, Rotation::cur());
///             let s = meta.query_selector(s);
///
///             // BUG: Should be a * b - c
///             Some(("buggy R1CS", s * (a * b + c)))
///         });
///
///         MyConfig { a, b, c, s }
///     }
///
///     fn synthesize(&self, config: MyConfig, mut layouter: impl Layouter<F>) -> Result<(), Error> {
///         layouter.assign_region(|| "Example region", |mut region| {
///             config.s.enable(&mut region, 0)?;
///             region.assign_advice(|| "a", config.a, 0, || {
///                 self.a.map(F::from)
///             })?;
///             region.assign_advice(|| "b", config.b, 0, || {
///                 self.b.map(F::from)
///             })?;
///             region.assign_advice(|| "c", config.c, 0, || {
///                 (self.a * self.b).map(F::from)
///             })?;
///             Ok(())
///         })
///     }
/// }
///
/// // Assemble the private inputs to the circuit.
/// let circuit = MyCircuit {
///     a: Value::known(2),
///     b: Value::known(4),
/// };
///
/// // This circuit has no public inputs.
/// let instance = vec![];
///
/// let prover = MockProver::<Fp>::run(K, &circuit, instance).unwrap();
/// assert_eq!(
///     prover.verify(),
///     Err(vec![VerifyFailure::ConstraintNotSatisfied {
///         constraint: ((0, "R1CS constraint").into(), 0, "buggy R1CS").into(),
///         location: FailureLocation::InRegion {
///             region: (0, "Example region").into(),
///             offset: 0,
///         },
///         cell_values: vec![
///             (((Any::Advice, 0).into(), 0).into(), "0x2".to_string()),
///             (((Any::Advice, 1).into(), 0).into(), "0x4".to_string()),
///             (((Any::Advice, 2).into(), 0).into(), "0x8".to_string()),
///         ],
///     }])
/// );
///
/// // If we provide a too-small K, we get an error.
/// assert!(matches!(
///     MockProver::<Fp>::run(2, &circuit, vec![]).unwrap_err(),
///     Error::NotEnoughRowsAvailable {
///         current_k,
///     } if current_k == 2,
/// ));
/// ``` 

/// --- START OF CUSTOM FUNCTIONALITY --- ///

/// A cell can be an advice, fixed, or instance cell
#[derive(PartialEq, Eq, Hash, Clone, PartialOrd, Ord)]
pub enum Cell {
    /// (col, row) for an advice cell
    Advice(usize, usize),

    /// (col, row) for an fixed cell
    Fixed(usize, usize),

    /// (col, row) for a instance cell
    Instance(usize, usize)
}

// Custom cell print format. Example: Advice col 1, row 2 is A1.2
impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Advice(arg0, arg1) => {
                write!(f, "A{}.{}", arg0, arg1)
            },
            Self::Fixed(arg0, arg1) => {
                write!(f, "F{}.{}", arg0, arg1)
            },
            Self::Instance(arg0, arg1) => {
                 write!(f, "I{}.{}", arg0, arg1)
            }
        }
    }
}

/// A cell set holds a collection from the circuit:
/// a. An instance/public input cell
/// b. An equality relation over multiple cells
/// c. An expression on cells that must equal 0 for a valid circuit
#[derive(PartialEq, Eq, Clone)]
pub enum CellSet<F: Field> {
    /// Instance cell with row and col
    Instance(usize, usize),

    /// List of cells assigned to be equal
    Equality(Vec<Cell>),

    /// List of cells in an expression, along with the expression
    Expr(Vec<Cell>, Vec<AbsExpression<F>>) 
}

// Custom cellset print format
impl<F: Field> fmt::Debug for CellSet<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Instance(arg0, arg1) => {
                write!(f, "Inst( I{}.{} )", arg0, arg1)
            }
            Self::Equality(v) => {
                write!(f, "Eq( {:?} )", v)
            }
            Self::Expr(v, e) => {
                write!(f, "Expr( {:?} : {:#?} )", v, e)
            }
        }
    }
}

// Custom hash function avoids hashing on the expression ast
impl<F: Field> Hash for CellSet<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            CellSet::Instance(c, r) => {
                state.write_u8(1);
                c.hash(state);
                r.hash(state);
            }
            CellSet::Equality(v, ) => {
                state.write_u8(2);
                v.hash(state);
            }
            CellSet::Expr(v, _) => {
                state.write_u8(3);
                v.hash(state);
            }
        }
    }
}

// Custom full order: instance < equality < expr. 
// Within instance sets, sort by col number then row number
// Within equality and expression sets, sort on the vec of Cells
impl<F: Field> Ord for CellSet<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self {
            CellSet::Instance(c, r) => {
                match other {
                    CellSet::Instance(co, ro) => {
                        let vec_other = vec![co, ro];
                        vec![c, r].cmp(&vec_other)
                    },
                    _ => Ordering::Less
                }
            },
            CellSet::Equality(v) => {
                match other {
                    CellSet::Instance(_, _) => Ordering::Greater,
                    CellSet::Equality(vo) => v.cmp(vo),
                    CellSet::Expr(_, _) => Ordering::Less
                }
            },
            CellSet::Expr(v, _) => {
                match other {
                    CellSet::Expr(vo, _) => v.cmp(vo),
                    _ => Ordering::Greater,
                }
            }
        }
    }
}

// Custom partial order uses full order
impl<F: Field> PartialOrd for CellSet<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Like the existing Expression enum in circuit.rs, except it uses my Cell enum
/// so that the expressions reference absolute cell locations instead of offsets
/// designed for gates that apply at every row
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AbsExpression<F: Field> {

    /// Constant polynomial (number?)
    Constant(F),

    /// Virtual selector. Selector(usize, bool), where usize is the index of the
    /// selector in the order it was added, and bool is true if it's a "simple
    /// selector" which has some additional restrictions
    Selector(Selector),

    /// Fixed cell at (fixed col, row)
    Fixed(usize, usize),

    /// Advice (witness) cell at (advice col, row)
    Advice(usize, usize),

    /// Instance cell at (instance col, row)
    Instance(usize, usize),

    /// Negated polynomial
    Negated(Box<AbsExpression<F>>),

    /// Sum of two polynomials
    Sum(Box<AbsExpression<F>>, Box<AbsExpression<F>>),

    /// Product of two polynomials
    Product(Box<AbsExpression<F>>, Box<AbsExpression<F>>),

    /// Scaled polynomial
    Scaled(Box<AbsExpression<F>>, F),
}

/// The MockProver object holds all the data about an instantiated circuit.
/// Heavily extended for Logos functionality
#[derive(Debug)]
pub struct MockProver<F: Field> {
    
    /// Original MockProver members ///
    
    /// as far as I can tell this just ensures that n is a power of 2
    pub k: u32,
    
    /// Number of rows in the circuit, n = 2^k 
    pub n: u32,

    /// The ConstraintSystem holds a lot of pertinent info, notably the gates
    /// and equality assignments
    pub cs: ConstraintSystem<F>,

    /// The regions in the circuit.
    pub regions: Vec<Region>,

    /// The current region being assigned to. Will be `None` after the circuit 
    /// has been synthesized.
    pub current_region: Option<Region>,

    /// The fixed cells in the circuit, arranged as [column][row].
    pub fixed: Vec<Vec<CellValue<F>>>,

    /// The advice cells in the circuit, arranged as [column][row].
    pub advice: Vec<Vec<CellValue<F>>>,

    /// The instance cells in the circuit, arranged as [column][row].
    pub instance: Vec<Vec<InstanceValue<F>>>,

    /// Seems to hold the T/F values of each selector, although could be the
    /// "simple" bool instead. Likely indexed [selector col][row]
    pub selectors: Vec<Vec<bool>>,

    /// Permutation holds several representations of cell equality constraints
    pub permutation: permutation::keygen::Assembly,

    /// A range of available rows for assignment and copies.
    pub usable_rows: Range<usize>,

    /// Added MockProver members ///

    /// Tracks what advice cells have been visited during initial DFS
    pub visited_advice: Vec<Vec<bool>>,
    
    /// Tracks what fixed cells have been visited during initial DFS
    pub visited_fixed: Vec<Vec<bool>>,
    
    /// Tracks what instance cells have been visited during initial DFS
    pub visited_instance: Vec<Vec<bool>>,

    /// Contains all the CellSets found in the circuit. HashSet to eliminate 
    /// redundant entries
    pub cellsets: HashSet<CellSet<F>>,

    /// self.cellsets is transferred here and sorted after initial DFS so that
    /// the CellSets can be sorted and referred to by index in later steps
    pub cellsets_vect: Vec<CellSet<F>>,

    /// Records the CellSets each advice cell is a part of
    /// tracker_advice[col][row] is a list of indices in cellsets_vect
    pub tracker_advice: Vec<Vec<Vec<usize>>>,
    
    /// Records the CellSets each instance cell is a part of. 
    pub tracker_instance: Vec<Vec<Vec<usize>>>,
    
    /// Records the CellSets each fixed cell is a part of.
    pub tracker_fixed: Vec<Vec<Vec<usize>>>,

}

/// PrintGraph has many functions that are used to construct a graph from the 
/// original MockProver object.
pub trait PrintGraph<F: Field> {
    

    /// Main function to build the graph. This is the entry point to do
    /// everything so far
    fn build_graph(&mut self);
    
    /// Runs DFS to collect all CellSets in the circuit. Branches to all Cells
    /// in an equality or expression with the parameter cell. 
    fn dfs(&mut self, cell: Cell);

    /// given a Cell enum, return a Vec of Cell enums representing all the cells
    /// constrained to be equal to the input. return[0] is the original input 
    /// Cell, return[1...n] are the new ones, if any. If the cell isn't in
    /// permutations, will also return just the input
    fn get_cells_in_group(&self, cell: Cell) -> Vec<Cell>;
    
    /// Given a cell, return all the gate instances it's a part of. The return
    /// include the gate's id, it's instantiated offset, and its non-zero
    /// expressions as AbsExprs
    fn get_gate_instances(&self, cell: Cell) -> Vec<(usize, i32, Vec<AbsExpression<F>>)>;

    /// Return a Vec of all the queried Cells in a gate instance, which is a 
    /// gate instance and offset
    fn get_cells_in_gate(&self, gate_ind: usize, offset: i32) -> Vec<Cell>;

    /// Look in self.permutation.columns for a Column with index equal to cell's
    /// column, and type equal to cell's type. Return the index of this Column
    /// if it exists, otherwise return -1 (Selector columns)
    fn get_perm_col(&self, cell: Cell) -> i32;

    /// Converts all Expressions in a gate instance into a Vec of AbsExpressions.
    /// If an expression simplifies to 0, it isn't included.
    fn get_abs_expressions(&self, gate: &Gate<F>, offset: i32) -> Vec<AbsExpression<F>>;

    /// Converts a halo2 Expression which uses VirtualCells into a Logos
    /// AbsExpression which represents expressions for gate instances, which use
    /// actual Cells that have a row instead of VirtualCells. Also recursively 
    /// simplifies the generated AbsExpr using Cell values. For example, 
    /// Prod(cell1, cell2) = cell2 if cell1 = 1
    fn get_abs_expression(&self, expr: &Expression<F>, offset: i32) -> AbsExpression<F>;

    /// Returns true if match_expr is a Constant equal to match_value or a Cell 
    /// that contains the value match_value
    fn evaluate_equal(&self, match_expr: &AbsExpression<F>, match_value: F) -> bool;

    /// Modify the the data structures necessary to perform one iteration of 
    /// BFS from the current cellset.
    fn bfs_manip(&self, tracker: &Vec<usize>, 
        visited_cellsets: &mut HashMap<usize, bool>, queue: &mut VecDeque<usize>, 
        edges: &mut Vec<Vec<usize>>, cellset_index: usize);

    /// print all the CellSets in sorted order. Only functional after calling 
    /// self.build_graph()
    fn print_cellsets(&self);

    /// print each tracking vector. Only functional after calling 
    /// self.build_graph()
    fn print_trackers(&self);

    // === NOT IN USE === //
    
    // constructs the full graph that grows from a single instance cell. for 
    // circuits with just one instance assignment, this is the only thing 
    // build_graph will return
    // fn build_graph_from_instance(&self, col: usize, row: usize);
}

impl<F: Field> PrintGraph<F> for MockProver<F> {
    
    // print all the CellSets in sorted order. Only functional after calling 
    // self.build_graph()
    fn print_cellsets(&self) {
        println!("SORTED CELLSET: \n {:#?}", self.cellsets_vect);
    }

    // print each tracking vector. Only functional after calling 
    // self.build_graph()
    fn print_trackers(&self) {
        println!("TRACKERS: \n");
        println!("Advice: {:#?}", self.tracker_advice);
        println!("Fixed: {:#?}", self.tracker_fixed);
        println!("Instance: {:#?}", self.tracker_instance);
    }

    // Main function to build the graph. This is the entry point to do
    // everything so far
    fn build_graph(&mut self) {    
        
        // Iterate through all instance cells (public inputs) as DFS start 
        // points because they are the public inputs, and the circuit's only 
        // guarenteed external access points.
        for (i, col) in self.instance.clone().iter().enumerate() {
            for (j, cell) in col.iter().enumerate() {
                
                // if the instance cell is assigned, add it to self.cellsets
                // and start a DFS. otherwise do nothing
                if let InstanceValue::Assigned(_) = cell {
                    self.cellsets.insert(CellSet::Instance(i, j));
                    let icell = Cell::Instance(i, j);
                    self.dfs(icell);
                }
            }
        }

        // self.cellsets served its purpose by eliminating redundant 
        // elements. Now use it to populate a vect and sort it
        for elem in &self.cellsets {
            self.cellsets_vect.push(elem.clone());
        }
        self.cellsets_vect.sort();

        // Record which cellsets (indexed in self.cellsets_vect) each cell is a 
        // member of in the self.tracker_... member variables
        for (i, cs) in self.cellsets_vect.iter().enumerate() {
            match cs {

                // If it's an Instance CellSet, then add i to 
                // tracker_instance[c][r]
                CellSet::Instance(col, row) => {
                    self.tracker_instance[*col][*row].push(i);
                }

                // Otherwise it's an Equality or Expr CellSet. So add each Cell 
                // in the set to tracker_<type>[c][r]
                CellSet::Equality(v)
                | CellSet::Expr(v, _) => {
                    for c in v.iter() {
                        match c {
                            Cell::Advice(col, row) => {
                                self.tracker_advice[*col][*row].push(i);
                            },
                            Cell::Fixed(col, row) => {
                                self.tracker_fixed[*col][*row].push(i);
                            }
                            Cell::Instance(col, row) => {
                                self.tracker_instance[*col][*row].push(i);
                            }
                        }
                    }
                }
            }
        }

        // Make dependency graph //

        // queue contains cells that need to be visited in a BFS that starts 
        // from the instance cells
        let mut queue: VecDeque<usize> = VecDeque::new();

        // visited_cellsets is a HashMap that records the visited state of each
        // cell. Every cell is in one of 3 situations in relation to the HM:
        //  1. not present - cell hasn't been seen. you can queue it and give it
        //     an edge
        //  2. present, false - cell is already in queue; you can't queue it but
        //     you can give it an edge
        //  3. present, true - cell has already been fully processed; you can't 
        //     queue it or give it an edge
        let mut visited_cellsets: HashMap<usize, bool> = HashMap::new();

        // edges is an adjacency list that stores directed edges between 
        // indexed CellSets.
        let mut edges: Vec<Vec<usize>> = vec![vec![]; self.cellsets_vect.len()];

        // Push Instance CellSets into queue. They start in state 2, so edges
        // can be drawn to them but they can't be added to the queue again.
        for i in 0..self.cellsets_vect.len() {
            if let CellSet::Instance(_, _) = self.cellsets_vect[i] {
                queue.push_back(i);
                visited_cellsets.insert(i, false);
            }
        }

        // BFS
        while !queue.is_empty() {

            // pop the front CellSet and set it to state 3
            let front = queue.pop_front().unwrap();
            *visited_cellsets.get_mut(&front).unwrap() = true;

            // visit the CellSet's neighbors. Behavior depends on CellSet type
            match &self.cellsets_vect[front] {

                // Instance CellSet. This shouldn't be reached because all of 
                // these should be queued at the start. But I guess just point
                // to it and add it to the queue
                CellSet::Instance(col, row) => {
                    self.bfs_manip(
                        &self.tracker_instance[*col][*row],
                        &mut visited_cellsets,
                        &mut queue,
                        &mut edges,
                        front
                    );
                },

                // Equality and Expr CellSets: iterate through all the component
                // cells, adding to the queue and adding edges as necessary
                CellSet::Equality(cell_vect) 
                | CellSet::Expr(cell_vect, _)=> {
                    for cell in cell_vect {
                        match cell {
                            Cell::Advice(col, row) => {
                                self.bfs_manip(
                                    &self.tracker_advice[*col][*row],
                                    &mut visited_cellsets,
                                    &mut queue,
                                    &mut edges,
                                    front
                                );
                            },
                            Cell::Fixed(col, row) => {
                                self.bfs_manip(
                                    &self.tracker_fixed[*col][*row],
                                    &mut visited_cellsets,
                                    &mut queue,
                                    &mut edges,
                                    front
                                );
                            },
                            Cell::Instance(col, row) => {
                                self.bfs_manip(
                                    &self.tracker_instance[*col][*row],
                                    &mut visited_cellsets,
                                    &mut queue,
                                    &mut edges,
                                    front
                                );
                            }
                        }
                    }
                },
            }
        }

        // Temporary result: construct Graphviz-compatible output and print
        let mut string = String::from("digraph G {\n");
        for (from_node, out_neighbors) in edges.iter().enumerate() {
            for to_node in out_neighbors {
                string += &"\"".to_string();
                string += &format!("{:#?}", self.cellsets_vect[from_node]);
                string += &"\"->\"".to_string();
                string += &format!("{:#?}", self.cellsets_vect[*to_node]);
                string += &"\"\n".to_string();
            }
        }
        string += &"}";

        println!("\n\n{}\n\n", string);
        println!("Edges = {:#?}", edges);
    }

    // Modify the the data structures necessary to perform one iteration of 
    // BFS from the current cellset.
    fn bfs_manip(&self, tracker: &Vec<usize>, 
        visited_cellsets: &mut HashMap<usize, bool>, queue: &mut VecDeque<usize>, 
        edges: &mut Vec<Vec<usize>>, cellset_index: usize) {
        
        for common_cellset in tracker {
            if !visited_cellsets.contains_key(&common_cellset) || !visited_cellsets[&common_cellset] {
                if !visited_cellsets.contains_key(&common_cellset) {
                    queue.push_back(*common_cellset);
                    visited_cellsets.insert(*common_cellset, false);
                }
                edges[cellset_index].push(*common_cellset);
            }
        }

    }

    // Runs DFS to collect all CellSets in the circuit. Branches to all Cells
    // in an equality or expression with the parameter cell. 
    fn dfs(&mut self, cell: Cell) {
        
        // Check if cell has been visited. If not, then mark it as visited in
        // the apropriate self.visited_<type> vec
        match cell {
            Cell::Advice(c, r) => {
                if self.visited_advice[c][r] { return }
                self.visited_advice[c][r] = true;
            },
            Cell::Fixed(c, r) => {
                if self.visited_fixed[c][r] { return }
                self.visited_fixed[c][r] = true;
            }
            Cell::Instance(c, r) => {
                if self.visited_instance[c][r] { return }
                self.visited_instance[c][r] = true;
            }
        }

        // Find all cells in an equality with the current cell and insert them
        // in self.cellsets. Then run DFS on all the equal cells
        let mut equalities = self.get_cells_in_group(cell.clone());
        equalities.sort();
        if equalities.len() > 1 {
            self.cellsets.insert(CellSet::Equality(equalities.clone()));
        }
        for e in equalities.iter() {
            self.dfs(e.clone());
        }

        // Find all gate instances that contain the current cell. Insert the
        // gate instances into self.cellsets. Then run DFS on all the other 
        // members of those instances.  
        let gates = self.get_gate_instances(cell.clone());
        for (gate_ind, offset, exprs) in gates {
            let mut gate_members = self.get_cells_in_gate(gate_ind, offset);
            gate_members.sort();
            //let mut sorted_gms = gate_members.clone();
            self.cellsets.insert(CellSet::Expr(gate_members.clone(), exprs));
            for gm in gate_members.iter() {
                self.dfs(gm.clone());
            }
        }
    }
    
    // given a Cell enum, return a Vec of Cell enums representing all the cells
    // constrained to be equal to the input. return[0] is the original input 
    // Cell, return[1...n] are the new ones, if any. If the cell isn't in
    // permutations, will also return just the input
    fn get_cells_in_group(&self, cell: Cell) -> Vec<Cell> {

        let mut cells = vec![cell.clone()];

        // Check if Cell::Type(col, row) maps to anything in 
        // self.permutation.columns. If not, just return the 1-element Vec.
        let perm_row = match cell {
            Cell::Advice(_, r) => r,
            Cell::Fixed(_, r) => r,
            Cell::Instance(_, r) => r
        };
        let perm_col = self.get_perm_col(cell);
        if perm_col == -1 {
            return cells;
        }
        
        // prepare to iterate through the equality mapping
        let perm_col = perm_col as usize;
        let mut cur_col = perm_col;
        let mut cur_row = perm_row;
        (cur_col, cur_row) = self.permutation.mapping[cur_col][cur_row];
        
        // The equality mapping is a loop, so go step-by-step until the start is
        // reached, adding each new Cell to the return Vec.  
        while (cur_col, cur_row) != (perm_col, perm_row) {
            let Column {
                index: c_ind, 
                column_type: c_type
            } = self.permutation.columns[cur_col];

            let next_cell = match c_type {
                Any::Fixed => Cell::Fixed(c_ind, cur_row),
                Any::Advice => Cell::Advice(c_ind, cur_row),
                Any::Instance => Cell::Instance(c_ind, cur_row)
            };
            cells.push(next_cell);
            (cur_col, cur_row) = self.permutation.mapping[cur_col][cur_row];
        }
        
        cells
    }

    // Given a cell, return all the gate instances it's a part of. The return
    // include the gate's id, it's instantiated offset, and its non-zero
    // expressions as AbsExprs
    fn get_gate_instances(&self, cell: Cell) -> Vec<(usize, i32, Vec<AbsExpression<F>>)> {
        
        // Get the halo2 Column associated with the cell, and extract its row
        let (col_obj, row) = match cell {
            Cell::Advice(c, r) => { 
                (Column {index: c, column_type: Any::Advice}, r) 
            },
            Cell::Fixed(c, r) => { 
                (Column {index: c, column_type: Any::Fixed}, r) 
            },
            Cell::Instance(c, r) => { 
                (Column {index: c, column_type: Any::Instance}, r) 
            } 
        };  

        // Build a list of gate instances
        let mut gate_instances: Vec<(usize, i32, Vec<AbsExpression<F>>)> = vec![];
        
        // Loop over all gates
        for (i, g) in self.cs.gates.iter().enumerate() {
            
            // Find the minimum and maximum rotations from 0 among all queried gates
            let mut max_rot = i32::MIN;
            let mut min_rot = i32::MAX;
            for vc in g.queried_cells().iter() {
                let Rotation(r) = vc.rotation;
                min_rot = min(min_rot, r);
                max_rot = max(max_rot, r);
            }

            // For each gate, go through all the queried cells and see if
            // there's an offset that allows the parameter gate to correspond
            // with the given queried cell. Also check that this offset doesn't
            // mean any other VirtualCells are instantiated out of bounds, and
            // that the gate is useful (has at least 1 non-zero AbsExpr).
            for vc in g.queried_cells().iter() {
                if vc.column == col_obj {

                    // calculate offset given that the parameter cell is in a
                    // given position. 
                    let offset = (row as i32) - vc.rotation.0;

                    // Ensure the min and max rotations are still in bounds
                    if self.usable_rows.start as i32 <= offset + min_rot 
                        && offset + max_rot < self.usable_rows.end as i32 {
                        
                        // Add the gate instance if it has non-zero AbsExprs
                        let abs_exprs = self.get_abs_expressions(g, offset);
                        if !abs_exprs.is_empty() {
                            gate_instances.push((i, offset, abs_exprs));
                        }
                    }
                }
            }
        }
        
        gate_instances
    }

    // Converts all Expressions in a gate instance into a Vec of AbsExpressions.
    // If an expression simplifies to 0, it isn't included.
    fn get_abs_expressions(&self, gate: &Gate<F>, offset: i32) -> Vec<AbsExpression<F>> {
        
        let mut abs_exprs = vec![];
        for expr in gate.polynomials() {
            let abs_expr = self.get_abs_expression(expr, offset);
            if !self.evaluate_equal(&abs_expr, F::ZERO) {
                abs_exprs.push(abs_expr);
            }
        }
        
        abs_exprs
    }

    // Converts a halo2 Expression which uses VirtualCells into a Logos
    // AbsExpression which represents expressions for gate instances, which use
    // actual Cells that have a row instead of VirtualCells. Also recursively 
    // simplifies the generated AbsExpr using Cell values. For example, 
    //Prod(cell1, cell2) = cell2 if cell1 = 1
    fn get_abs_expression(&self, expr: &Expression<F>, offset: i32) -> AbsExpression<F> {
        match expr {

            // Directly map Constants, Selectors, Negations, and Scales to AbsExprs
            Expression::Constant(f) => {
                AbsExpression::Constant(*f)
            }
            Expression::Selector(s) => {
                AbsExpression::Selector(*s)
            }
            Expression::Negated(boxed_expr) => {
                let abs_expr = self.get_abs_expression(&**boxed_expr, offset);
                AbsExpression::Negated(Box::new(abs_expr))
            }
            Expression::Scaled(boxed_expr, scale) => {
                AbsExpression::Scaled(Box::new(self.get_abs_expression(boxed_expr, offset)), *scale)
            }

            // Replace Fixed, Advice, and Instance relative cells with absolute
            // rows using the desired offset
            Expression::Fixed(FixedQuery {index: _, column_index, rotation} ) => {
                AbsExpression::Fixed(*column_index, offset as usize + rotation.0 as usize)
            }
            Expression::Advice(AdviceQuery {index: _, column_index, rotation}) => {
                AbsExpression::Advice(*column_index, offset as usize + rotation.0 as usize)
            }
            Expression::Instance(InstanceQuery {index: _, column_index, rotation}) => {
                AbsExpression::Instance(*column_index, offset as usize + rotation.0 as usize)
            }

            // Recursively convert the two addends in a Sum, and simplify
            Expression::Sum(boxed1, boxed2) => {
                let abs_expr_1 = self.get_abs_expression(&**boxed1, offset);
                let abs_expr_2 = self.get_abs_expression(&**boxed2, offset);
                
                // sum(0, x) = x
                if self.evaluate_equal(&abs_expr_1, F::ZERO) {
                    return abs_expr_2
                }

                // sum(x, 0) = x
                if self.evaluate_equal(&abs_expr_2, F::ZERO) {
                    return abs_expr_1
                }

                AbsExpression::Sum(Box::new(abs_expr_1), Box::new(abs_expr_2))
            }

            // Recursively convert the two factors in a Product, and simplify
            Expression::Product(boxed1, boxed2) => {
                let abs_expr_1 = self.get_abs_expression(&**boxed1, offset);
                let abs_expr_2 = self.get_abs_expression(&**boxed2, offset);

                // product(0, x) = 0
                if self.evaluate_equal(&abs_expr_1, F::ZERO) {
                    return abs_expr_1
                }

                // product(x, 0) = 0
                if self.evaluate_equal(&abs_expr_2, F::ZERO) {
                    return abs_expr_2
                }

                // product(1, x) = x
                if self.evaluate_equal(&abs_expr_1, F::ONE) {
                    return abs_expr_2
                }

                // product(x, 1) = x
                if self.evaluate_equal(&abs_expr_2, F::ONE) {
                    return abs_expr_1
                }

                AbsExpression::Product(Box::new(abs_expr_1), Box::new(abs_expr_2))
            }
        }
    }

    // Returns true if match_expr is a Constant equal to match_value or a Cell 
    // that contains the value match_value
    fn evaluate_equal(&self, match_expr: &AbsExpression<F>, match_value: F) -> bool {
        match match_expr {
            AbsExpression::Constant(c) => {
                *c == match_value
            } 
            AbsExpression::Advice(c, r) => {
                self.advice[*c][*r] == CellValue::Assigned(match_value)
            }
            AbsExpression::Fixed(c, r) => {
                self.fixed[*c][*r] == CellValue::Assigned(match_value) 
            }
            AbsExpression::Instance(c, r) => {
                self.instance[*c][*r] == InstanceValue::Assigned(match_value)
            }
            _ => false
        }
    }

    // Return a Vec of all the queried Cells in a gate instance, which is a 
    // gate instance and offset
    fn get_cells_in_gate(&self, gate_ind: usize, offset: i32) -> Vec<Cell> {

        let mut cells = HashSet::new();
        
        // iterate over all the VirtualCells in queried_cells()
        for VirtualCell{
                column: Column {
                    index: vc_ind, 
                    column_type: c_type
                }, 
                rotation: Rotation(r)
            } in self.cs.gates[gate_ind].queried_cells() {
            
            // use the rotation (relative row offset) of the abstract gate added
            // to the offset to get the VirtualCell's absolute position for this
            // gate instance, and add it to the Vec
            match c_type {
                Any::Advice => {
                    cells.insert(Cell::Advice(*vc_ind, offset as usize + *r as usize));
                }
                Any::Fixed => {
                    cells.insert(Cell::Fixed(*vc_ind, offset as usize + *r as usize));
                }
                Any::Instance => {
                    cells.insert(Cell::Instance(*vc_ind, offset as usize + *r as usize));
                }
            }
        }
        
        Vec::from_iter(cells)
    }
    

    /// Look in self.permutation.columns for a Column with index equal to cell's
    /// column, and type equal to cell's type. Return the index of this Column
    /// if it exists, otherwise return -1 (Selector columns)
    fn get_perm_col(&self, cell: Cell) -> i32 {

        // Go through each column in self.permuation.columns and see if it
        // matches the column type and index of cell
        for (pi, pcol) in self.permutation.columns.iter().enumerate() {
            match cell {
                Cell::Advice(c, _) => {
                    if pcol.column_type == Any::Advice && pcol.index == c {
                        return pi as i32;
                    }
                }
                Cell::Fixed(c, _) => {
                    if pcol.column_type == Any::Fixed && pcol.index == c {
                        return pi as i32;
                    }
                }
                Cell::Instance(c, _) => {
                    if pcol.column_type == Any::Instance && pcol.index == c {
                        return pi as i32;
                    }
                }
            }
        }

        // returns -1 if the column isn't found
        -1
    }
}

/// --- END OF CUSTOM FUNCTIONALITY --- ///

///
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstanceValue<F: Field> {
    ///
    Assigned(F),
    ///
    Padding,
}

impl<F: Field> InstanceValue<F> {
    fn value(&self) -> F {
        match self {
            InstanceValue::Assigned(v) => *v,
            InstanceValue::Padding => F::ZERO,
        }
    }
}

impl<F: Field> Assignment<F> for MockProver<F> {
    fn enter_region<NR, N>(&mut self, name: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        assert!(self.current_region.is_none());
        self.current_region = Some(Region {
            name: name().into(),
            columns: HashSet::default(),
            rows: None,
            enabled_selectors: HashMap::default(),
            cells: vec![],
        });
    }

    fn exit_region(&mut self) {
        self.regions.push(self.current_region.take().unwrap());
    }

    fn enable_selector<A, AR>(&mut self, _: A, selector: &Selector, row: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        // Track that this selector was enabled. We require that all selectors are enabled
        // inside some region (i.e. no floating selectors).
        self.current_region
            .as_mut()
            .unwrap()
            .enabled_selectors
            .entry(*selector)
            .or_default()
            .push(row);

        self.selectors[selector.0][row] = true;

        Ok(())
    }

    fn query_instance(
        &self,
        column: Column<Instance>,
        row: usize,
    ) -> Result<circuit::Value<F>, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.instance
            .get(column.index())
            .and_then(|column| column.get(row))
            .map(|v| circuit::Value::known(v.value()))
            .ok_or(Error::BoundsFailure)
    }

    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Advice>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> circuit::Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        if let Some(region) = self.current_region.as_mut() {
            region.update_extent(column.into(), row);
            region.cells.push((column.into(), row));
        }

        *self
            .advice
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? =
            CellValue::Assigned(to().into_field().evaluate().assign()?);

        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Fixed>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> circuit::Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        if let Some(region) = self.current_region.as_mut() {
            region.update_extent(column.into(), row);
            region.cells.push((column.into(), row));
        }

        *self
            .fixed
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? =
            CellValue::Assigned(to().into_field().evaluate().assign()?);

        Ok(())
    }

    fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), crate::plonk::Error> {
        if !self.usable_rows.contains(&left_row) || !self.usable_rows.contains(&right_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.permutation
            .copy(left_column, left_row, right_column, right_row)
    }

    fn fill_from_row(
        &mut self,
        col: Column<Fixed>,
        from_row: usize,
        to: circuit::Value<Assigned<F>>,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&from_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        for row in self.usable_rows.clone().skip(from_row) {
            self.assign_fixed(|| "", col, row, || to)?;
        }

        Ok(())
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // TODO: Do something with namespaces :)
    }

    fn pop_namespace(&mut self, _: Option<String>) {
        // TODO: Do something with namespaces :)
    }
}

impl<F: Field + Ord> MockProver<F> {
    /// Runs a synthetic keygen-and-prove operation on the given circuit, collecting data
    /// about the constraints and their assignments.
    pub fn run<ConcreteCircuit: Circuit<F>>(
        k: u32,
        circuit: &ConcreteCircuit,
        instance: Vec<Vec<F>>,
    ) -> Result<Self, Error> {
        let n = 1 << k;

        let mut cs = ConstraintSystem::default();
        let config = ConcreteCircuit::configure(&mut cs);
        let cs = cs;

        if n < cs.minimum_rows() {
            return Err(Error::not_enough_rows_available(k));
        }

        if instance.len() != cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }

        let instance = instance
            .into_iter()
            .map(|instance| {
                if instance.len() > n - (cs.blinding_factors() + 1) {
                    return Err(Error::InstanceTooLarge);
                }

                let mut instance_values = vec![InstanceValue::Padding; n];
                for (idx, value) in instance.into_iter().enumerate() {
                    instance_values[idx] = InstanceValue::Assigned(value);
                }

                Ok(instance_values)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Fixed columns contain no blinding factors.
        let fixed = vec![vec![CellValue::Unassigned; n]; cs.num_fixed_columns];
        let selectors = vec![vec![false; n]; cs.num_selectors];
        // Advice columns contain blinding factors.
        let blinding_factors = cs.blinding_factors();
        let usable_rows = n - (blinding_factors + 1);
        let advice = vec![
            {
                let mut column = vec![CellValue::Unassigned; n];
                // Poison unusable rows.
                for (i, cell) in column.iter_mut().enumerate().skip(usable_rows) {
                    *cell = CellValue::Poison(i);
                }
                column
            };
            cs.num_advice_columns
        ];
        let permutation = permutation::keygen::Assembly::new(n, &cs.permutation);
        let constants = cs.constants.clone();
        
        let visited_advice = vec![vec![false; n]; advice.len()];
        let visited_fixed = vec![vec![false; n]; fixed.len()];
        let visited_instance = vec![vec![false; n]; instance.len()];

        let tracker_advice = vec![vec![vec![]; n]; advice.len()];
        let tracker_fixed = vec![vec![vec![]; n]; fixed.len()];
        let tracker_instance = vec![vec![vec![]; n]; instance.len()];

        let mut prover = MockProver {
            k,
            n: n as u32,
            cs,
            regions: vec![],
            current_region: None,
            fixed,
            advice,
            instance,
            selectors,
            permutation,
            usable_rows: 0..usable_rows,
            visited_advice,
            visited_fixed,
            visited_instance,
            cellsets: HashSet::new(),
            cellsets_vect: Vec::new(),
            tracker_advice,
            tracker_fixed,
            tracker_instance,
        };

        println!("before synthesize in dev.rs");
        ConcreteCircuit::FloorPlanner::synthesize(&mut prover, circuit, config, constants)?;
        println!("after synthesize in dev.rs");
        let (cs, selector_polys) = prover.cs.compress_selectors(prover.selectors.clone());
        prover.cs = cs;
        prover.fixed.extend(selector_polys.into_iter().map(|poly| {
            let mut v = vec![CellValue::Unassigned; n];
            for (v, p) in v.iter_mut().zip(&poly[..]) {
                *v = CellValue::Assigned(*p);
            }
            v
        }));

        Ok(prover)
    }

    /// Returns `Ok(())` if this `MockProver` is satisfied, or a list of errors indicating
    /// the reasons that the circuit is not satisfied.
    pub fn verify(&self) -> Result<(), Vec<VerifyFailure>> {
        let n = self.n as i32;

        // Check that within each region, all cells used in instantiated gates have been
        // assigned to.
        let selector_errors = self.regions.iter().enumerate().flat_map(|(r_i, r)| {
            r.enabled_selectors.iter().flat_map(move |(selector, at)| {
                // Find the gates enabled by this selector
                self.cs
                    .gates
                    .iter()
                    // Assume that if a queried selector is enabled, the user wants to use the
                    // corresponding gate in some way.
                    //
                    // TODO: This will trip up on the reverse case, where leaving a selector
                    // un-enabled keeps a gate enabled. We could alternatively require that
                    // every selector is explicitly enabled or disabled on every row? But that
                    // seems messy and confusing.
                    .enumerate()
                    .filter(move |(_, g)| g.queried_selectors().contains(selector))
                    .flat_map(move |(gate_index, gate)| {
                        at.iter().flat_map(move |selector_row| {
                            // Selectors are queried with no rotation.
                            let gate_row = *selector_row as i32;

                            gate.queried_cells().iter().filter_map(move |cell| {
                                // Determine where this cell should have been assigned.
                                let cell_row = ((gate_row + n + cell.rotation.0) % n) as usize;

                                match cell.column.column_type() {
                                    Any::Instance => {
                                        // Handle instance cells, which are not in the region.
                                        let instance_value =
                                            &self.instance[cell.column.index()][cell_row];
                                        match instance_value {
                                            InstanceValue::Assigned(_) => None,
                                            _ => Some(VerifyFailure::InstanceCellNotAssigned {
                                                gate: (gate_index, gate.name()).into(),
                                                region: (r_i, r.name.clone()).into(),
                                                gate_offset: *selector_row,
                                                column: cell.column.try_into().unwrap(),
                                                row: cell_row,
                                            }),
                                        }
                                    }
                                    _ => {
                                        // Check that it was assigned!
                                        if r.cells.contains(&(cell.column, cell_row)) {
                                            None
                                        } else {
                                            Some(VerifyFailure::CellNotAssigned {
                                                gate: (gate_index, gate.name()).into(),
                                                region: (r_i, r.name.clone()).into(),
                                                gate_offset: *selector_row,
                                                column: cell.column,
                                                offset: cell_row as isize
                                                    - r.rows.unwrap().0 as isize,
                                            })
                                        }
                                    }
                                }
                            })
                        })
                    })
            })
        });

        // Check that all gates are satisfied for all rows.
        let gate_errors =
            self.cs
                .gates
                .iter()
                .enumerate()
                .flat_map(|(gate_index, gate)| {
                    // We iterate from n..2n so we can just reduce to handle wrapping.
                    (n..(2 * n)).flat_map(move |row| {
                        gate.polynomials().iter().enumerate().filter_map(
                            move |(poly_index, poly)| match poly.evaluate(
                                &|scalar| Value::Real(scalar),
                                &|_| panic!("virtual selectors are removed during optimization"),
                                &util::load(n, row, &self.cs.fixed_queries, &self.fixed),
                                &util::load(n, row, &self.cs.advice_queries, &self.advice),
                                &util::load_instance(
                                    n,
                                    row,
                                    &self.cs.instance_queries,
                                    &self.instance,
                                ),
                                &|a| -a,
                                &|a, b| a + b,
                                &|a, b| a * b,
                                &|a, scalar| a * scalar,
                            ) {
                                Value::Real(x) if x.is_zero_vartime() => None,
                                Value::Real(_) => Some(VerifyFailure::ConstraintNotSatisfied {
                                    constraint: (
                                        (gate_index, gate.name()).into(),
                                        poly_index,
                                        gate.constraint_name(poly_index),
                                    )
                                        .into(),
                                    location: FailureLocation::find_expressions(
                                        &self.cs,
                                        &self.regions,
                                        (row - n) as usize,
                                        Some(poly).into_iter(),
                                    ),
                                    cell_values: util::cell_values(
                                        gate,
                                        poly,
                                        &util::load(n, row, &self.cs.fixed_queries, &self.fixed),
                                        &util::load(n, row, &self.cs.advice_queries, &self.advice),
                                        &util::load_instance(
                                            n,
                                            row,
                                            &self.cs.instance_queries,
                                            &self.instance,
                                        ),
                                    ),
                                }),
                                Value::Poison => Some(VerifyFailure::ConstraintPoisoned {
                                    constraint: (
                                        (gate_index, gate.name()).into(),
                                        poly_index,
                                        gate.constraint_name(poly_index),
                                    )
                                        .into(),
                                }),
                            },
                        )
                    })
                });

        // Check that all lookups exist in their respective tables.
        let lookup_errors =
            self.cs
                .lookups
                .iter()
                .enumerate()
                .flat_map(|(lookup_index, lookup)| {
                    let load = |expression: &Expression<F>, row| {
                        expression.evaluate(
                            &|scalar| Value::Real(scalar),
                            &|_| panic!("virtual selectors are removed during optimization"),
                            &|query| {
                                let query = self.cs.fixed_queries[query.index];
                                let column_index = query.0.index();
                                let rotation = query.1 .0;
                                self.fixed[column_index]
                                    [(row as i32 + n + rotation) as usize % n as usize]
                                    .into()
                            },
                            &|query| {
                                let query = self.cs.advice_queries[query.index];
                                let column_index = query.0.index();
                                let rotation = query.1 .0;
                                self.advice[column_index]
                                    [(row as i32 + n + rotation) as usize % n as usize]
                                    .into()
                            },
                            &|query| {
                                let query = self.cs.instance_queries[query.index];
                                let column_index = query.0.index();
                                let rotation = query.1 .0;
                                Value::Real(
                                    self.instance[column_index]
                                        [(row as i32 + n + rotation) as usize % n as usize]
                                        .value(),
                                )
                            },
                            &|a| -a,
                            &|a, b| a + b,
                            &|a, b| a * b,
                            &|a, scalar| a * scalar,
                        )
                    };

                    assert!(lookup.table_expressions.len() == lookup.input_expressions.len());
                    assert!(self.usable_rows.end > 0);

                    // We optimize on the basis that the table might have been filled so that the last
                    // usable row now has the fill contents (it doesn't matter if there was no filling).
                    // Note that this "fill row" necessarily exists in the table, and we use that fact to
                    // slightly simplify the optimization: we're only trying to check that all input rows
                    // are contained in the table, and so we can safely just drop input rows that
                    // match the fill row.
                    let fill_row: Vec<_> = lookup
                        .table_expressions
                        .iter()
                        .map(move |c| load(c, self.usable_rows.end - 1))
                        .collect();

                    // In the real prover, the lookup expressions are never enforced on
                    // unusable rows, due to the (1 - (l_last(X) + l_blind(X))) term.
                    let mut table: Vec<Vec<_>> = self
                        .usable_rows
                        .clone()
                        .filter_map(|table_row| {
                            let t = lookup
                                .table_expressions
                                .iter()
                                .map(move |c| load(c, table_row))
                                .collect();

                            if t != fill_row {
                                Some(t)
                            } else {
                                None
                            }
                        })
                        .collect();
                    table.sort_unstable();

                    let mut inputs: Vec<(Vec<_>, usize)> = self
                        .usable_rows
                        .clone()
                        .filter_map(|input_row| {
                            let t = lookup
                                .input_expressions
                                .iter()
                                .map(move |c| load(c, input_row))
                                .collect();

                            if t != fill_row {
                                // Also keep track of the original input row, since we're going to sort.
                                Some((t, input_row))
                            } else {
                                None
                            }
                        })
                        .collect();
                    inputs.sort_unstable();

                    let mut i = 0;
                    inputs
                        .iter()
                        .filter_map(move |(input, input_row)| {
                            while i < table.len() && &table[i] < input {
                                i += 1;
                            }
                            if i == table.len() || &table[i] > input {
                                assert!(table.binary_search(input).is_err());

                                Some(VerifyFailure::Lookup {
                                    lookup_index,
                                    location: FailureLocation::find_expressions(
                                        &self.cs,
                                        &self.regions,
                                        *input_row,
                                        lookup.input_expressions.iter(),
                                    ),
                                })
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                });

        // Check that permutations preserve the original values of the cells.
        let perm_errors = {
            // Original values of columns involved in the permutation.
            let original = |column, row| {
                self.cs
                    .permutation
                    .get_columns()
                    .get(column)
                    .map(|c: &Column<Any>| match c.column_type() {
                        Any::Advice => self.advice[c.index()][row],
                        Any::Fixed => self.fixed[c.index()][row],
                        Any::Instance => {
                            let cell: &InstanceValue<F> = &self.instance[c.index()][row];
                            CellValue::Assigned(cell.value())
                        }
                    })
                    .unwrap()
            };

            // Iterate over each column of the permutation
            self.permutation
                .mapping
                .iter()
                .enumerate()
                .flat_map(move |(column, values)| {
                    // Iterate over each row of the column to check that the cell's
                    // value is preserved by the mapping.
                    values.iter().enumerate().filter_map(move |(row, cell)| {
                        let original_cell = original(column, row);
                        let permuted_cell = original(cell.0, cell.1);
                        if original_cell == permuted_cell {
                            None
                        } else {
                            let columns = self.cs.permutation.get_columns();
                            let column = columns.get(column).unwrap();
                            Some(VerifyFailure::Permutation {
                                column: (*column).into(),
                                location: FailureLocation::find(
                                    &self.regions,
                                    row,
                                    Some(column).into_iter().cloned().collect(),
                                ),
                            })
                        }
                    })
                })
        };

        let mut errors: Vec<_> = iter::empty()
            .chain(selector_errors)
            .chain(gate_errors)
            .chain(lookup_errors)
            .chain(perm_errors)
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            // Remove any duplicate `ConstraintPoisoned` errors (we check all unavailable
            // rows in case the trigger is row-specific, but the error message only points
            // at the constraint).
            errors.dedup_by(|a, b| match (a, b) {
                (
                    a @ VerifyFailure::ConstraintPoisoned { .. },
                    b @ VerifyFailure::ConstraintPoisoned { .. },
                ) => a == b,
                _ => false,
            });
            Err(errors)
        }
    }

    /// Panics if the circuit being checked by this `MockProver` is not satisfied.
    ///
    /// Any verification failures will be pretty-printed to stderr before the function
    /// panics.
    ///
    /// Apart from the stderr output, this method is equivalent to:
    /// ```ignore
    /// assert_eq!(prover.verify(), Ok(()));
    /// ```
    pub fn assert_satisfied(&self) {
        if let Err(errs) = self.verify() {
            for err in errs {
                err.emit(self);
                eprintln!();
            }
            panic!("circuit was not satisfied");
        }
    }
}

#[cfg(test)]
mod tests {
    use group::ff::Field;
    use pasta_curves::Fp;

    use super::{FailureLocation, MockProver, VerifyFailure};
    use crate::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            Advice, Any, Circuit, Column, ConstraintSystem, Error, Expression, Selector,
            TableColumn,
        },
        poly::Rotation,
    };

    #[test]
    fn unassigned_cell() {
        const K: u32 = 4;

        #[derive(Clone)]
        struct FaultyCircuitConfig {
            a: Column<Advice>,
            q: Selector,
        }

        struct FaultyCircuit {}

        impl Circuit<Fp> for FaultyCircuit {
            type Config = FaultyCircuitConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
                let a = meta.advice_column();
                let b = meta.advice_column();
                let q = meta.selector();

                meta.create_gate("Equality check", |cells| {
                    let a = cells.query_advice(a, Rotation::prev());
                    let b = cells.query_advice(b, Rotation::cur());
                    let q = cells.query_selector(q);

                    // If q is enabled, a and b must be assigned to.
                    vec![q * (a - b)]
                });

                FaultyCircuitConfig { a, q }
            }

            fn without_witnesses(&self) -> Self {
                Self {}
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fp>,
            ) -> Result<(), Error> {
                layouter.assign_region(
                    || "Faulty synthesis",
                    |mut region| {
                        // Enable the equality gate.
                        config.q.enable(&mut region, 1)?;

                        // Assign a = 0.
                        region.assign_advice(|| "a", config.a, 0, || Value::known(Fp::ZERO))?;

                        // BUG: Forget to assign b = 0! This could go unnoticed during
                        // development, because cell values default to zero, which in this
                        // case is fine, but for other assignments would be broken.
                        Ok(())
                    },
                )
            }
        }

        let prover = MockProver::run(K, &FaultyCircuit {}, vec![]).unwrap();
        assert_eq!(
            prover.verify(),
            Err(vec![VerifyFailure::CellNotAssigned {
                gate: (0, "Equality check").into(),
                region: (0, "Faulty synthesis".to_owned()).into(),
                gate_offset: 1,
                column: Column::new(1, Any::Advice),
                offset: 1,
            }])
        );
    }

    #[test]
    fn bad_lookup() {
        const K: u32 = 4;

        #[derive(Clone)]
        struct FaultyCircuitConfig {
            a: Column<Advice>,
            q: Selector,
            table: TableColumn,
        }

        struct FaultyCircuit {}

        impl Circuit<Fp> for FaultyCircuit {
            type Config = FaultyCircuitConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
                let a = meta.advice_column();
                let q = meta.complex_selector();
                let table = meta.lookup_table_column();

                meta.lookup(|cells| {
                    let a = cells.query_advice(a, Rotation::cur());
                    let q = cells.query_selector(q);

                    // If q is enabled, a must be in the table.
                    // When q is not enabled, lookup the default value instead.
                    let not_q = Expression::Constant(Fp::one()) - q.clone();
                    let default = Expression::Constant(Fp::from(2));
                    vec![(q * a + not_q * default, table)]
                });

                FaultyCircuitConfig { a, q, table }
            }

            fn without_witnesses(&self) -> Self {
                Self {}
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fp>,
            ) -> Result<(), Error> {
                layouter.assign_table(
                    || "Doubling table",
                    |mut table| {
                        (1..(1 << (K - 1)))
                            .map(|i| {
                                table.assign_cell(
                                    || format!("table[{}] = {}", i, 2 * i),
                                    config.table,
                                    i - 1,
                                    || Value::known(Fp::from(2 * i as u64)),
                                )
                            })
                            .fold(Ok(()), |acc, res| acc.and(res))
                    },
                )?;

                layouter.assign_region(
                    || "Good synthesis",
                    |mut region| {
                        // Enable the lookup on rows 0 and 1.
                        config.q.enable(&mut region, 0)?;
                        config.q.enable(&mut region, 1)?;

                        // Assign a = 2 and a = 6.
                        region.assign_advice(
                            || "a = 2",
                            config.a,
                            0,
                            || Value::known(Fp::from(2)),
                        )?;
                        region.assign_advice(
                            || "a = 6",
                            config.a,
                            1,
                            || Value::known(Fp::from(6)),
                        )?;

                        Ok(())
                    },
                )?;

                layouter.assign_region(
                    || "Faulty synthesis",
                    |mut region| {
                        // Enable the lookup on rows 0 and 1.
                        config.q.enable(&mut region, 0)?;
                        config.q.enable(&mut region, 1)?;

                        // Assign a = 4.
                        region.assign_advice(
                            || "a = 4",
                            config.a,
                            0,
                            || Value::known(Fp::from(4)),
                        )?;

                        // BUG: Assign a = 5, which doesn't exist in the table!
                        region.assign_advice(
                            || "a = 5",
                            config.a,
                            1,
                            || Value::known(Fp::from(5)),
                        )?;

                        Ok(())
                    },
                )
            }
        }

        let prover = MockProver::run(K, &FaultyCircuit {}, vec![]).unwrap();
        assert_eq!(
            prover.verify(),
            Err(vec![VerifyFailure::Lookup {
                lookup_index: 0,
                location: FailureLocation::InRegion {
                    region: (2, "Faulty synthesis").into(),
                    offset: 1,
                }
            }])
        );
    }
}
