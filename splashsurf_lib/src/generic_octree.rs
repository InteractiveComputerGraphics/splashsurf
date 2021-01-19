use crate::ThreadSafe;
use arrayvec::ArrayVec;
use rayon::{Scope, ScopeFifo};
use std::collections::VecDeque;
use std::ops::DerefMut;

#[derive(Clone, Debug, Default)]
pub struct GenericOctree<T> {
    root: GenericOctreeNode<T>,
}

#[derive(Clone, Debug, Default)]
pub struct GenericOctreeNode<T> {
    children: ArrayVec<[Box<GenericOctreeNode<T>>; 8]>,
    data: T,
}

impl<T> GenericOctree<T> {
    pub fn new(root_data: T) -> Self {
        Self {
            root: GenericOctreeNode::new(root_data),
        }
    }

    pub fn root(&self) -> &GenericOctreeNode<T> {
        &self.root
    }

    /// Iterator that yields all nodes of the tree in depth-first order.
    pub fn dfs_iter(&self) -> impl Iterator<Item = &GenericOctreeNode<T>> {
        let mut queue = Vec::new();
        queue.push(&self.root);

        let iter = move || -> Option<&GenericOctreeNode<T>> {
            if let Some(next_node) = queue.pop() {
                // Enqueue all children
                queue.extend(
                    next_node
                        .children()
                        .iter()
                        .rev()
                        .map(std::ops::Deref::deref),
                );

                Some(next_node)
            } else {
                None
            }
        };

        std::iter::from_fn(iter)
    }

    /// Visits a node and its children in breadth-first order. The visitor is applied before enqueuing each node's children.
    pub fn visit_mut_bfs<F: FnMut(&mut GenericOctreeNode<T>)>(&mut self, mut visitor: F) {
        let mut queue_down = VecDeque::new();
        queue_down.push_back(&mut self.root);

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

impl<T: ThreadSafe> GenericOctree<T> {
    /// Visits a node and its children in breadth-first order. The visitor is applied before enqueuing each node's children. Parallel version.
    pub fn visit_mut_bfs_par<F: Fn(&mut GenericOctreeNode<T>) + ThreadSafe>(&mut self, visitor: F) {
        let v = &visitor;
        rayon::scope_fifo(move |s| self.root.visit_mut_bfs_par(s, v));
    }

    /// Visits a node and its children in depth-first post-order. The visitor is applied after processing each node's children. Parallel version.
    pub fn visit_mut_dfs_post_par<F: Fn(&mut GenericOctreeNode<T>) + ThreadSafe>(
        &mut self,
        visitor: F,
    ) {
        let v = &visitor;
        rayon::scope(move |s| self.root.visit_mut_dfs_post_par(s, v));
    }
}

impl<T> GenericOctreeNode<T> {
    pub fn new(data: T) -> Self {
        Self {
            children: Default::default(),
            data,
        }
    }

    pub fn children(&self) -> &[Box<GenericOctreeNode<T>>] {
        self.children.as_slice()
    }

    pub fn children_mut(&mut self) -> &mut [Box<GenericOctreeNode<T>>] {
        self.children.as_mut_slice()
    }

    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T: ThreadSafe> GenericOctreeNode<T> {
    fn visit_mut_bfs_par<'scope, F: Fn(&mut Self) + ThreadSafe>(
        &'scope mut self,
        s: &ScopeFifo<'scope>,
        visitor: &'scope F,
    ) {
        visitor(self);

        // Spawn tasks for all children
        for child in self.children.iter_mut() {
            s.spawn_fifo(move |s| child.visit_mut_bfs_par(s, visitor));
        }
    }

    fn visit_mut_dfs_post_par<'scope, F: Fn(&mut Self) + ThreadSafe>(
        &'scope mut self,
        _s: &Scope<'scope>,
        visitor: &'scope F,
    ) {
        // Create a new scope to ensure that tasks are completed before the visitor runs
        rayon::scope(|s| {
            for child in self.children.iter_mut() {
                s.spawn(move |s| child.visit_mut_dfs_post_par(s, visitor));
            }
        });

        visitor(self);
    }
}
