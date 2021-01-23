use rayon::{Scope, ScopeFifo};
use std::collections::VecDeque;
use std::iter::FusedIterator;
use std::ops::{Deref, DerefMut};

pub trait TreeNode {
    /// Returns a slice of all child nodes
    fn children(&self) -> &[Box<Self>];
    /// Returns a mutable slice of all child nodes
    fn children_mut(&mut self) -> &mut [Box<Self>];
}

pub trait VisitableTree: TreeNode {
    /// An iterator over all nodes and its children in depth-first order.
    fn dfs_iter<'a>(&'a self) -> DfsIter<'a, Self> {
        DfsIter::new(self)
    }

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

    /// An iterator over all nodes and its children in breadth-first order.
    fn bfs_iter<'a>(&'a self) -> BfsIter<'a, Self> {
        BfsIter::new(self)
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

impl<'a, T: TreeNode + ?Sized> FusedIterator for DfsIter<'a, T> {}

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

impl<'a, T: TreeNode + ?Sized> FusedIterator for BfsIter<'a, T> {}

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
            T: TreeNode + Send + Sync + ?Sized,
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
            T: TreeNode + Send + Sync + ?Sized,
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
}

impl<T: TreeNode> VisitableTree for T {}
impl<T: TreeNode + Send + Sync> ParVisitableTree for T {}
