use crate::ThreadSafe;
use rayon::{Scope, ScopeFifo};
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};

pub trait TreeNode {
    /// Returns a slice of all child nodes
    fn children(&self) -> &[Box<Self>];
    /// Returns a mutable slice of all child nodes
    fn children_mut(&mut self) -> &mut [Box<Self>];
}

pub trait VisitableTree: TreeNode {
    /// Visits a node and its children in depth-first order.
    fn visit_dfs<'a, F: FnMut(&'a Self)>(&'a self, mut visitor: F) {
        let mut queue_down = Vec::new();
        queue_down.push(self);

        while let Some(current_node) = queue_down.pop() {
            visitor(current_node);
            queue_down.extend(current_node.children().iter().rev().map(Deref::deref));
        }
    }

    /// Visits a node and its children in depth-first order. The visitor is applied before enqueuing each node's children.
    fn visit_mut_dfs<F: FnMut(&mut Self)>(&mut self, mut visitor: F) {
        let mut queue_down = Vec::new();
        queue_down.push(self);

        while let Some(current_node) = queue_down.pop() {
            visitor(current_node);
            queue_down.extend(
                current_node
                    .children_mut()
                    .iter_mut()
                    .rev()
                    .map(DerefMut::deref_mut),
            );
        }
    }

    /// Visits a node and its children in breadth-first order.
    fn visit_bfs<'a, F: FnMut(&'a Self)>(&'a self, mut visitor: F) {
        let mut queue_down = VecDeque::new();
        queue_down.push_back(self);

        while let Some(current_node) = queue_down.pop_front() {
            visitor(current_node);
            queue_down.extend(current_node.children().iter().map(Deref::deref));
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

pub trait ParVisitableTree: TreeNode {
    /// Visits a node and its children in breadth-first order. The visitor is applied before enqueuing each node's children. Parallel version.
    fn visit_mut_bfs_par<F: Fn(&mut Self) + ThreadSafe>(&mut self, visitor: F)
    where
        Self: ThreadSafe,
    {
        let v = &visitor;
        rayon::scope_fifo(move |s| self.visit_mut_bfs_par_impl(s, v));
    }

    /// Visits a node and its children in depth-first post-order. The visitor is applied after processing each node's children. Parallel version.
    fn visit_mut_dfs_post_par<F: Fn(&mut Self) + ThreadSafe>(&mut self, visitor: F)
    where
        Self: ThreadSafe,
    {
        let v = &visitor;
        rayon::scope(move |s| self.visit_mut_dfs_post_par_impl(s, v));
    }

    fn visit_mut_bfs_par_impl<'scope, F: Fn(&mut Self) + ThreadSafe>(
        &'scope mut self,
        s: &ScopeFifo<'scope>,
        visitor: &'scope F,
    ) where
        Self: ThreadSafe,
    {
        visitor(self);

        // Spawn tasks for all children
        for child in self.children_mut().iter_mut() {
            s.spawn_fifo(move |s| child.visit_mut_bfs_par_impl(s, visitor));
        }
    }

    fn visit_mut_dfs_post_par_impl<'scope, F: Fn(&mut Self) + ThreadSafe>(
        &'scope mut self,
        _s: &Scope<'scope>,
        visitor: &'scope F,
    ) where
        Self: ThreadSafe,
    {
        // Create a new scope to ensure that tasks are completed before the visitor runs
        rayon::scope(|s| {
            for child in self.children_mut().iter_mut() {
                s.spawn(move |s| child.visit_mut_dfs_post_par_impl(s, visitor));
            }
        });

        visitor(self);
    }
}

impl<T: TreeNode> VisitableTree for T {}
impl<T: TreeNode + ThreadSafe> ParVisitableTree for T {}
