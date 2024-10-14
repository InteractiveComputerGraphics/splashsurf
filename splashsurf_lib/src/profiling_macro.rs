//! Provides the [`profile`](crate::profile) macro or a dummy implementation depending on the selected feature

#[cfg(feature = "profiling")]
/// Creates a scope for profiling
///
/// This macro provides profiling inspired by the [`coarse-prof`](https://crates.io/crates/coarse-prof) crate.
/// The biggest difference to `coarse-prof` is that it supports profiling in multi-threaded programs.
///
/// Note that this macro only works when the `profiling` feature is enabled. Otherwise a dummy
/// implementation is provided with the same rules that expand to nothing. This allows the user to
/// disable this coarse grained profiling without modifying code depending on this macro. This can
/// be helpful to minimize overhead when using more elaborate profiling approaches.
///
/// Profiling works using scope guards that increment a thread local [`Profiler`](crate::profiling::Profiler)
/// (stored in the static [`PROFILER`](static@crate::profiling::PROFILER) variable) when they are dropped.
/// To evaluate the collected timings, the [`write`](crate::profiling::write) function can be used.
///
/// The [`write`](crate::profiling::write) function produces a human readable, hierarchically
/// structured overview of the gathered profiling data, like this:
/// ```text
/// frame: 100.00%, 10.40ms/call @ 96.17Hz
///   physics: 3.04%, 3.16ms/call @ 9.62Hz
///     collisions: 33.85%, 1.07ms/call @ 9.62Hz
///   render: 96.84%, 10.07ms/call @ 96.17Hz
/// ```
/// It collects the timings over all threads and accumulates total runtime and number of calls for
/// scopes at the same point in the call graph. As it is not easily possible to automatically connect
/// the graph across thread boundaries, this is possible manually using one of the macro rules
/// explained below.
///
/// The `profile!` macro locally returns a scope guard tracking the time instant when it was invoked.
/// Note that this guard is stored in a variable called `_profiling_scope_guard` which can result
/// in shadowing.
///
/// Note that even though it is safe to transfer a scope guard across threads, this does lead to
/// inconsistent timings when it is dropped. So this is most certainly not what you want to do.
/// Transferring [`ScopeId`](crate::profiling::ScopeId)s across thread boundaries however does make
/// sense when a to manually assign a parent scope across threads.
///
/// There are four ways to use this macro.
///  1. A simple scope with a name. The nesting of this scope in the profiling hierarchy is inferred from the
///     scope that is surrounding the new scope on the current thread.
///     ```ignore
///     {
///         profile!("new scope")
///     }
///     ```
///  2. A simple scope with a name and an id. An id of a new scope can be stored explicitly in a variable
///     named by the user (`scope_id` here). This can be used by other scopes to manually assign it a
///     parent scope in the profiling hierarchy.
///     ```ignore
///     {
///         profile!(scope_id, "new scope")
///     }
///     ```
///  3. A scope with a manually specified parent scope. As the profiling macros cannot automatically infer
///     the hierarchy across threads, it is possible to manually assign a scope to a parent scope.
///     ```ignore
///     {
///         profile!(scope_id, "outer scope")
///         {
///             vec.par_iter().for_each(|item| {
///                 profile!("inner scope", parent = scope_id);
///             });
///         }
///     }
///     ```
///  4. The second and third option can be combined:
///     ```ignore
///     profile!(inner_id, "inner scope", parent = outer_id);
///     ```
#[macro_export]
#[cfg_attr(docsrs, doc(cfg(feature = "profiling")))]
macro_rules! profile {
    ($name:expr) => {
        use $crate::profile_impl;
        profile_impl!($name);
    };
    ($scope_id:ident, $name:expr) => {
        use $crate::profile_impl;
        profile_impl!($scope_id, $name);
    };
    ($name:expr, parent = $parent_id:ident) => {
        use $crate::profile_impl;
        profile_impl!($name, parent = $parent_id);
    };
    ($scope_id:ident, $name:expr, parent = $parent_id:ident) => {
        use $crate::profile_impl;
        profile_impl!($scope_id, $name, parent = $parent_id);
    };
}

#[cfg(not(feature = "profiling"))]
/// No-op macro if profiling is disabled
#[macro_export]
macro_rules! profile {
    ($name:expr) => {};
    ($scope_id:ident, $name:expr) => {};
    ($name:expr, parent = $parent_id:ident) => {};
    ($scope_id:ident, $name:expr, parent = $parent_id:ident) => {};
}
