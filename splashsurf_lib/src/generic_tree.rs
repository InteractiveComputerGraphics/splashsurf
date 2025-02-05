//! Generic tree visitation functions that can be used with tree-like structures
//!
//! In order to use these algorithms, the data structure has to implement the [`TreeNode`] and for
//! mutable access the [`TreeNodeMut`] trait. These traits only require to give access to a slice
//! of child nodes of a node.
//!
//! This module provides algorithms for breadth-first and depth-first visitation.
//! Overview of the traits:
//!  - [`VisitableTree`] provides non-mutable sequential iteration.
//!  - [`MutVisitableTree`] provides sequential visitation using a visitor function with mutable access to the current node.
//!  - [`ParVisitableTree`] provides parallel visitation using a visitor function, parallelized using rayon.
//!  - [`ParMutVisitableTree`] provides parallel visitation using a visitor function with mutable access to the current node, parallelized using rayon.
//!
//! Note that the mutation of nodes during visitation is safe as the mutation is only possible either
//! before the children are enqueued or after the children were already processed.
//!

use parking_lot::RwLock;
use rayon::{Scope, ScopeFifo};
use std::collections::VecDeque;
use std::iter::FusedIterator;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

// TODO: Tests for the algorithms

/// Trait that has to be implemented by tree-like structures to make them visitable
pub trait TreeNode {
    /// Returns a slice of all child nodes
    fn children(&self) -> &[Box<Self>];
}

/// Trait that has to be implemented by tree-like structures to visit and mutate them
pub trait TreeNodeMut: TreeNode {
    /// Returns a mutable slice of all child nodes
    fn children_mut(&mut self) -> &mut [Box<Self>];
}

/// Trait for non-mutable sequential tree iteration algorithms. Automatically implemented for types that implement [`TreeNode`].
pub trait VisitableTree: TreeNode {
    /// An iterator over all nodes and its children in depth-first order.
    fn dfs_iter(&self) -> DfsIter<'_, Self> {
        DfsIter::new(self)
    }

    /// An iterator over all nodes and its children in breadth-first order.
    fn bfs_iter(&self) -> BfsIter<'_, Self> {
        BfsIter::new(self)
    }
}

/// Trait for sequential tree visitation algorithms that support mutation during visitation. Automatically implemented for types that implement [`TreeNodeMut`].
pub trait MutVisitableTree: TreeNodeMut {
    /// Visits a node and its children in depth-first order. The visitor is applied before enqueuing each node's children.
    fn visit_mut_dfs<F: FnMut(&mut Self)>(&mut self, mut visitor: F) {
        let mut stack = Vec::new();
        stack.push(self);

        while let Some(current_node) = stack.pop() {
            visitor(current_node);
            stack.extend(
                current_node
                    .children_mut()
                    .iter_mut()
                    .rev()
                    .map(DerefMut::deref_mut),
            );
        }
    }

    /// Visits a node and its children in breadth-first order. The visitor is applied before enqueuing each node's children.
    fn visit_mut_bfs<F: FnMut(&mut Self)>(&mut self, mut visitor: F) {
        let mut queue_down = VecDeque::new();
        queue_down.push_back(self);

        while let Some(current_node) = queue_down.pop_front() {
            visitor(current_node);
            queue_down.extend(
                current_node
                    .children_mut()
                    .iter_mut()
                    .map(DerefMut::deref_mut),
            );
        }
    }
}

/// Depth-first search iterator returned by the [`VisitableTree::dfs_iter`] function
pub struct DfsIter<'a, T: ?Sized> {
    stack: Vec<&'a T>,
}

impl<'a, T: ?Sized> DfsIter<'a, T> {
    fn new(start: &'a T) -> Self {
        Self { stack: vec![start] }
    }
}

impl<'a, T: TreeNode + ?Sized> Iterator for DfsIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current_node) = self.stack.pop() {
            self.stack
                .extend(current_node.children().iter().rev().map(Deref::deref));
            Some(current_node)
        } else {
            None
        }
    }
}

impl<T: TreeNode + ?Sized> FusedIterator for DfsIter<'_, T> {}

/// Breadth-first search iterator returned by the [`VisitableTree::bfs_iter`] function
pub struct BfsIter<'a, T: ?Sized> {
    queue: VecDeque<&'a T>,
}

impl<'a, T: ?Sized> BfsIter<'a, T> {
    fn new(start: &'a T) -> Self {
        Self {
            queue: vec![start].into(),
        }
    }
}

impl<'a, T: TreeNode + ?Sized> Iterator for BfsIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current_node) = self.queue.pop_front() {
            self.queue
                .extend(current_node.children().iter().rev().map(Deref::deref));
            Some(current_node)
        } else {
            None
        }
    }
}

impl<T: TreeNode + ?Sized> FusedIterator for BfsIter<'_, T> {}

/// Trait for non-mutable parallel tree visitation algorithms. Automatically implemented for types that implement [`TreeNode`] and [`ThreadSafe`](crate::ThreadSafe).
pub trait ParVisitableTree: TreeNode {
    /// Visits a node and its children in breadth-first order. The visitor is applied in parallel to processing the children.
    fn par_visit_bfs<F>(&self, visitor: F)
    where
        Self: Sync,
        F: Fn(&Self) + Sync,
    {
        // Parallel implementation of recursive breadth-first visitation
        fn par_visit_bfs_impl<'scope, T, F>(
            node: &'scope T,
            s: &ScopeFifo<'scope>,
            visitor: &'scope F,
        ) where
            T: TreeNode + Sync + ?Sized,
            F: Fn(&T) + Sync,
        {
            // Spawn task for visitor
            s.spawn_fifo(move |_| visitor(node));

            // Spawn tasks for all children
            for child in node.children().iter().map(Deref::deref) {
                s.spawn_fifo(move |s| par_visit_bfs_impl(child, s, visitor));
            }
        }

        let v = &visitor;
        rayon::scope_fifo(move |s| par_visit_bfs_impl(self, s, v));
    }

    /// Visits a node and its children in breadth-first order, stops visitation on first error and returns it. The visitor is applied in parallel to processing the children.
    fn try_par_visit_bfs<E, F>(&self, visitor: F) -> Result<(), E>
    where
        Self: Sync,
        E: Send + Sync,
        F: Fn(&Self) -> Result<(), E> + Sync,
    {
        let error = Arc::new(RwLock::new(Ok(())));

        // Parallel implementation of recursive breadth-first visitation
        fn try_par_visit_bfs_impl<'scope, T, E, F>(
            node: &'scope T,
            s: &ScopeFifo<'scope>,
            error: Arc<RwLock<Result<(), E>>>,
            visitor: &'scope F,
        ) where
            T: TreeNode + Sync + ?Sized,
            E: Send + Sync + 'scope,
            F: Fn(&T) -> Result<(), E> + Sync,
        {
            // Stop recursion if there is already an error
            if error.read().is_err() {
                return;
            }

            // Spawn task for visitor
            {
                let error = error.clone();
                s.spawn_fifo(move |_| {
                    // Only run visitor if there was no error in the meantime
                    if error.read().is_ok() {
                        // Run visitor and check returned result
                        let res = visitor(node);
                        if res.is_err() {
                            let mut error_guard = error.write();
                            // Don't overwrite error if there is already one
                            if !error_guard.is_err() {
                                *error_guard = res;
                            }
                        }
                    }
                });
            }

            // Spawn tasks for all children
            for child in node.children().iter().map(Deref::deref) {
                let error = error.clone();
                s.spawn_fifo(move |s| try_par_visit_bfs_impl(child, s, error, visitor));
            }
        }

        // Start the visitation
        {
            let v = &visitor;
            let e = error.clone();
            rayon::scope_fifo(move |s| try_par_visit_bfs_impl(self, s, e, v));
        }

        // Return any potential error collected during visitation
        if !error.read().is_ok() {
            match Arc::try_unwrap(error) {
                Ok(e) => e.into_inner(),
                Err(_) => panic!("Unable to unwrap Arc that stores error of tree visitation"),
            }
        } else {
            Ok(())
        }
    }
}

/// Trait for mutable parallel tree visitation algorithms. Automatically implemented for types that implement [`TreeNodeMut`] and [`ThreadSafe`](crate::ThreadSafe).
pub trait ParMutVisitableTree: TreeNodeMut {
    /// Visits a node and its children in breadth-first order. The visitor is applied before enqueuing each node's children. Parallel version.
    fn par_visit_mut_bfs<F>(&mut self, visitor: F)
    where
        Self: Send + Sync,
        F: Fn(&mut Self) + Sync,
    {
        // Parallel implementation of recursive breadth-first visitation
        fn par_visit_mut_bfs_impl<'scope, T, F>(
            node: &'scope mut T,
            s: &ScopeFifo<'scope>,
            visitor: &'scope F,
        ) where
            T: TreeNodeMut + Send + Sync + ?Sized,
            F: Fn(&mut T) + Sync,
        {
            // Apply visitor before enqueuing children
            visitor(node);

            // Spawn tasks for all children
            for child in node.children_mut().iter_mut().map(DerefMut::deref_mut) {
                s.spawn_fifo(move |s| par_visit_mut_bfs_impl(child, s, visitor));
            }
        }

        let v = &visitor;
        rayon::scope_fifo(move |s| par_visit_mut_bfs_impl(self, s, v));
    }

    /// Visits a node and its children in depth-first post-order. The visitor is applied after processing each node's children. Parallel version.
    fn par_visit_mut_dfs_post<F>(&mut self, visitor: F)
    where
        Self: Send + Sync,
        F: Fn(&mut Self) + Sync,
    {
        fn par_visit_mut_dfs_post_impl<'scope, T, F>(
            node: &'scope mut T,
            _s: &Scope<'scope>,
            visitor: &'scope F,
        ) where
            T: TreeNodeMut + Send + Sync + ?Sized,
            F: Fn(&mut T) + Sync,
        {
            // Create a new scope to ensure that tasks are completed before the visitor runs
            rayon::scope(|s| {
                for child in node.children_mut().iter_mut().map(DerefMut::deref_mut) {
                    s.spawn(move |s| par_visit_mut_dfs_post_impl(child, s, visitor));
                }
            });

            visitor(node);
        }

        let v = &visitor;
        rayon::scope(move |s| par_visit_mut_dfs_post_impl(self, s, v));
    }

    /// Visits a node and its children in depth-first post-order, stops visitation on first error and returns it. The visitor is applied after processing each node's children. Parallel version.
    fn try_par_visit_mut_dfs_post<E, F>(&mut self, visitor: F) -> Result<(), E>
    where
        Self: Send + Sync,
        E: Send + Sync,
        F: Fn(&mut Self) -> Result<(), E> + Sync,
    {
        let error = Arc::new(RwLock::new(Ok(())));

        fn try_par_visit_mut_dfs_post_impl<'scope, T, E, F>(
            node: &'scope mut T,
            _s: &Scope<'scope>,
            error: Arc<RwLock<Result<(), E>>>,
            visitor: &'scope F,
        ) where
            T: TreeNodeMut + Send + Sync + ?Sized,
            E: Send + Sync,
            F: Fn(&mut T) -> Result<(), E> + Sync,
        {
            // Stop recursion if there is already an error
            if error.read().is_err() {
                return;
            }

            // Create a new scope to ensure that tasks are completed before the visitor runs
            rayon::scope(|s| {
                for child in node.children_mut().iter_mut().map(DerefMut::deref_mut) {
                    let error = error.clone();
                    s.spawn(move |s| try_par_visit_mut_dfs_post_impl(child, s, error, visitor));
                }
            });

            // Only run visitor if none of the child nodes returned an error
            if error.read().is_ok() {
                // Run visitor and check returned result
                let res = visitor(node);
                if res.is_err() {
                    let mut error_guard = error.write();
                    // Don't overwrite error if there is already one
                    if !error_guard.is_err() {
                        *error_guard = res;
                    }
                }
            }
        }

        // Start the visitation
        {
            let v = &visitor;
            let e = error.clone();
            rayon::scope(move |s| try_par_visit_mut_dfs_post_impl(self, s, e, v));
        }

        // Return any potential error collected during visitation
        if !error.read().is_ok() {
            match Arc::try_unwrap(error) {
                Ok(e) => e.into_inner(),
                Err(_) => panic!("Unable to unwrap Arc that stores error of tree visitation"),
            }
        } else {
            Ok(())
        }
    }
}

impl<T: TreeNode> VisitableTree for T {}
impl<T: TreeNodeMut> MutVisitableTree for T {}
impl<T: TreeNode + Send + Sync> ParVisitableTree for T {}
impl<T: TreeNodeMut + Send + Sync> ParMutVisitableTree for T {}
