//! Implementation details for the [`profile`](crate::profile) macro

use parking_lot::RwLock;
use std::collections::hash_map::RandomState;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::hash::{BuildHasher, Hash};
use std::io;
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use thread_local::ThreadLocal;

/// Thread local storage of the [`Profiler`]s storing all `Scope`s of the thread
pub static PROFILER: LazyLock<ThreadLocal<RwLock<Profiler>>> = LazyLock::new(ThreadLocal::new);

/// `RandomState` used to obtain `Hasher`s to hash [`ScopeId`]s for parent/child identification
pub static RANDOM_STATE: LazyLock<RandomState> = LazyLock::new(RandomState::new);

/// Implementation of the profile macro, use [`profile`](crate::profile) instead
#[doc(hidden)]
#[macro_export]
macro_rules! profile_impl {
    ($name:expr) => {
        let (_profiling_scope_guard, _) = $crate::profiling::PROFILER
            .get_or(Default::default)
            .write()
            .enter($name);
    };
    ($scope_id:ident, $name:expr) => {
        let (_profiling_scope_guard, $scope_id) = $crate::profiling::PROFILER
            .get_or(Default::default)
            .write()
            .enter($name);
    };
    ($name:expr, parent = $parent_id:ident) => {
        let (_profiling_scope_guard, _) = $crate::profiling::PROFILER
            .get_or(Default::default)
            .write()
            .enter_with_parent($name, &$parent_id);
    };
    ($scope_id:ident, $name:expr, parent = $parent_id:ident) => {
        let (_profiling_scope_guard, $scope_id) = $crate::profiling::PROFILER
            .get_or(Default::default)
            .write()
            .enter_with_parent($name, &$parent_id);
    };
}

/// A scope guard recording the elapsed time of the scope
pub struct Guard {
    enter_time: Instant,
}

impl Guard {
    fn new() -> Self {
        Self {
            enter_time: Instant::now(),
        }
    }
}

/// Dropping a `Guard` will add its recorded elapsed time to the top of the current local scope stack
impl Drop for Guard {
    fn drop(&mut self) {
        let duration = self.enter_time.elapsed();
        PROFILER
            .get()
            .expect("Missing thread local profiler")
            .write()
            .leave(duration)
    }
}

#[derive(Clone, Debug)]
struct Scope {
    name: &'static str,
    num_calls: usize,
    duration_sum: Duration,
    first_call: Instant,
}

impl Scope {
    fn new(name: &'static str) -> Self {
        Scope {
            name,
            num_calls: 0,
            duration_sum: Default::default(),
            first_call: Instant::now(),
        }
    }

    fn merge(&mut self, other: &Self) {
        if other.name == self.name {
            self.num_calls += other.num_calls;
            self.duration_sum += other.duration_sum;
            if other.first_call < self.first_call {
                self.first_call = other.first_call;
            }
        }
    }
}

/// Type used to uniquely identify scopes across threads
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub struct ScopeId {
    name: &'static str,
    parent_hash: u64,
}

impl ScopeId {
    fn get_hash(id: Option<&ScopeId>) -> u64 {
        RANDOM_STATE.hash_one(id)
    }
}

/// Profiler storing all locally created scopes with their timings and the current stack of scopes
#[derive(Default)]
pub struct Profiler {
    /// All local scopes that were created using the `enter` functions
    scopes: HashMap<ScopeId, Scope>,
    /// Current stack hierarchy of the local scopes
    scope_stack: Vec<ScopeId>,
    /// All local scopes that have no parent (or which have a manually assigned parent)
    roots: HashSet<ScopeId>,
}

impl Profiler {
    /// Resets all profiling data of this profiler
    pub fn reset(&mut self) {
        self.scopes.clear();
        self.scope_stack.clear();
        self.roots.clear();
    }

    /// Enter a scope with the given name and push it onto the stack of scopes. It will be a child scope of the current top of the stack.
    pub fn enter(&mut self, name: &'static str) -> (Guard, ScopeId) {
        // TODO: If the scope on top of the stack has the same name, use the next one as parent, to avoid huge chains for recursive functions
        let id = self.new_id(name, self.scope_stack.last());
        self.enter_with_id(name, id)
    }

    /// Enter a scope with the given name and push it onto the stack of scopes. It will be manually assigned as a child of the given parent scope.
    pub fn enter_with_parent(&mut self, name: &'static str, parent: &ScopeId) -> (Guard, ScopeId) {
        let id = self.new_id(name, Some(parent));
        self.enter_with_id(name, id)
    }

    fn enter_with_id(&mut self, name: &'static str, id: ScopeId) -> (Guard, ScopeId) {
        self.scopes.entry(id).or_insert_with(|| Scope::new(name));

        // Insert as root, even it has a manually assigned parent to prevent child scopes from ending up as roots themselves
        if self.scope_stack.is_empty() {
            self.roots.insert(id);
        }

        self.scope_stack.push(id);
        (Guard::new(), id)
    }

    /// Leave the scope at the top of the stack and increment its scope by the given duration
    fn leave(&mut self, duration: Duration) {
        if let Some(id) = self.scope_stack.pop() {
            if let Some(scope) = self.scopes.get_mut(&id) {
                scope.num_calls += 1;
                scope.duration_sum += duration;
            }
        }
    }

    fn new_id(&self, name: &'static str, parent: Option<&ScopeId>) -> ScopeId {
        ScopeId {
            name,
            parent_hash: ScopeId::get_hash(parent),
        }
    }
}

fn write_recursively<W: io::Write>(
    out: &mut W,
    sorted_scopes: &[(ScopeId, Scope)],
    current: &(ScopeId, Scope),
    parent_duration: Option<Duration>,
    depth: usize,
    is_parallel: bool,
) -> io::Result<()> {
    let (id, scope) = current;

    for _ in 0..depth {
        write!(out, "  ")?;
    }

    let duration_sum_secs = scope.duration_sum.as_secs_f64();
    let parent_duration_secs = parent_duration.map_or(duration_sum_secs, |t| t.as_secs_f64());
    let percent = duration_sum_secs / parent_duration_secs * 100.0;

    writeln!(
        out,
        "{}: {}{:3.2}%, {:>4.2}ms avg, {} {} (total: {:.3}s)",
        scope.name,
        if is_parallel { "≈" } else { "" },
        percent,
        duration_sum_secs * 1000.0 / (scope.num_calls as f64),
        scope.num_calls,
        if scope.num_calls > 1 { "calls" } else { "call" },
        duration_sum_secs
    )?;

    // Compute sum of runtimes of children
    let mut children_runtime = Duration::default();
    let current_hash = ScopeId::get_hash(Some(id));
    for s in sorted_scopes {
        let (child_id, child_scope) = s;
        if child_id.parent_hash == current_hash {
            children_runtime += child_scope.duration_sum;
        }
    }

    // If children were run in parallel, their total runtime can be larger than that of the current scope
    let children_parallel = children_runtime > scope.duration_sum;
    let own_runtime = if children_parallel {
        children_runtime
    } else {
        scope.duration_sum
    };

    // Process children in sorted order
    let current_hash = ScopeId::get_hash(Some(id));
    for s in sorted_scopes {
        let (child_id, _) = s;
        if child_id.parent_hash == current_hash {
            // TODO: Prevent infinite recursion for recursive functions, maybe remove current scope from map?
            //  Maybe we don't have this problem, instead it will be a huge chain which is as long as the recursion depth...
            write_recursively(
                out,
                sorted_scopes,
                s,
                Some(own_runtime),
                depth + 1,
                children_parallel,
            )?;
        }
    }

    Ok(())
}

/// Pretty print the collected profiling data of all thread local [`Profiler`]s to the given writer
pub fn write<W: io::Write>(out: &mut W) -> io::Result<()> {
    let mut merged_scopes = HashMap::<ScopeId, Scope>::new();
    let mut roots = HashSet::<ScopeId>::new();

    // Collect scopes over all threads
    for profiler in PROFILER.iter() {
        let profiler = profiler.read();
        roots.extend(profiler.roots.iter());

        for (&id, scope) in &profiler.scopes {
            merged_scopes
                .entry(id)
                .and_modify(|s| s.merge(scope))
                .or_insert_with(|| scope.clone());
        }
    }

    // Sort and filter root scopes
    let sorted_roots = {
        let root_hash = ScopeId::get_hash(None);
        let mut roots = roots
            .into_iter()
            // Remove roots that are not actual roots (happens if their parent was set manually)
            .filter(|id| id.parent_hash == root_hash)
            // Get (id, scope) tuple
            .flat_map(|id| merged_scopes.get(&id).cloned().map(|s| (id, s)))
            .collect::<Vec<_>>();

        roots.sort_unstable_by_key(|(_, s)| s.first_call);
        roots
    };

    // Sort all scopes by first call time
    let sorted_scopes = {
        let mut scopes = merged_scopes.into_iter().collect::<Vec<_>>();
        scopes.sort_unstable_by_key(|(_, s)| s.first_call);
        scopes
    };

    // Print the stats
    for root in &sorted_roots {
        write_recursively(out, sorted_scopes.as_slice(), root, None, 0, false)?;
    }

    Ok(())
}

/// Returns the pretty printed output of the collected profiling data as a `String`
pub fn write_to_string() -> Result<String, Box<dyn Error>> {
    let mut buffer = Vec::new();
    write(&mut buffer)?;
    Ok(String::from_utf8(buffer)?)
}

/// Resets the profiling data of all thread local [`Profiler`]s
///
/// Note that it should be ensure that this is called outside of any scopes. It's safe to also call
/// this function from inside a scope but it can lead to inconsistent profiling data (if a new scope
/// is created afterwards followed by dropping of an old scope).
pub fn reset() {
    for profiler in PROFILER.iter() {
        profiler.write().reset();
    }
}
