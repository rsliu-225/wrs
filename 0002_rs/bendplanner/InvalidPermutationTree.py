from treelib import Tree, Node
import copy


class IPTree:
    def __init__(self, num, name="invalid_tree"):
        self.node_ids = list(range(num))

        self.name = name
        self.tree = Tree(identifier=self.name)
        self.root = Node(tag="root", identifier="root", data=None)
        self.tree.add_node(self.root)

    def show(self):
        self.tree.show()

    def add_invalid_seq(self, seq=None, cache_idx=None, cache_data=None):
        if not seq:
            return

        str_seq = [str(i) for i in seq]

        for idx in list(range(len(seq))):
            if seq[idx] > self.node_ids[-1]:
                raise Exception(f"node {seq[idx]} exceeds the tree limit")

            node_id = "-" + '-'.join(str_seq[:idx + 1])
            parent_node_id = node_id[:-len(str_seq[idx]) - 1]
            parent_node_id = parent_node_id if parent_node_id else "root"

            if cache_idx is not None and idx == cache_idx:
                node_data = cache_data
            else:
                node_data = None

            if not self.tree.get_node(node_id):
                self.tree.create_node(
                    seq[idx], node_id, parent_node_id, node_data,
                )

    def whether_valid(self, seq=None):
        """
        Whether the sequence is valid, if the sequence is valid but part of it exists in the invalid tree,
        return the latest data stored node id as well as the latest data stored index.
        """
        if not seq:
            return False, (-1, None)

        node_id = ""
        cur = None
        latest_cache_idx = -1
        latest_cache_node_id = None

        for i, idx in enumerate(seq):
            node_id += '-' + str(idx)
            cur = self.tree.get_node(node_id)

            if not cur:
                return True, (latest_cache_idx, latest_cache_node_id)
            else:
                if cur.data is not None:
                    latest_cache_idx = i
                    latest_cache_node_id = node_id

            if cur.is_leaf():
                return False, (-1, None)

        if not cur:
            return False, (-1, None)
        else:
            remained_valid_children = len(self.node_ids) - len(seq) - \
                                      len(cur.successors(self.name))

            if remained_valid_children > 0:
                return True, (latest_cache_idx, latest_cache_node_id)
            else:
                return False, (-1, None)

    def get_potential_valid(self, pre_seq=None):
        # Use DFS to get the first potential valid and tuple of its cache index and cached data
        available_child_ids = self.node_ids[::-1]
        pre_node_ids = ""
        cache_tuple = (-1, None)

        if pre_seq:
            is_valid, cache_tuple = self.whether_valid(pre_seq)
            if is_valid:
                for idx in pre_seq:
                    available_child_ids.remove(idx)
                    pre_node_ids += '-' + str(idx)
            else:
                print("Invalid sequence!")
                return [], (-1, None)

        stack = [(pre_node_ids + '-' + str(idx), [x for x in available_child_ids if x != idx], cache_tuple)
                 for idx in available_child_ids]

        while stack:
            # print(stack)
            node_id, child_ids, latest_cache_tuple = stack.pop()
            # print(node_id, child_ids)
            seq_invalid = self.tree.get_node(node_id) and self.tree.get_node(node_id).is_leaf()
            if not seq_invalid:
                if not child_ids:
                    cache_data_tuple = (
                        latest_cache_tuple[0],
                        self.tree.get_node(latest_cache_tuple[1]).data if latest_cache_tuple[1] else None,
                    )
                    return [int(x) for x in node_id.split('-')[1:]], cache_data_tuple

                for idx in child_ids:
                    cur_node = self.tree.get_node(node_id)
                    if cur_node and cur_node.data is not None:
                        cur_cache_tuple = ((len(self.node_ids) - len(child_ids) - 1), node_id)
                    else:
                        cur_cache_tuple = latest_cache_tuple[:]

                    stack.append((node_id + '-' + str(idx), [x for x in child_ids if x != idx], cur_cache_tuple))

        print("Can not find any potentially valid sequence!")
        return [], None


if __name__ == '__main__':
    tree = IPTree(15)
    tree.add_invalid_seq([0])
    tree.add_invalid_seq([1, 0, 2, 3], 2, {2: 2})
    tree.add_invalid_seq([1, 3, 4], 1, {1: 1})
    tree.add_invalid_seq([1, 2, 3])
    tree.add_invalid_seq([1, 2, 4])
    tree.tree.show()

    print(tree.get_potential_valid([3]))
    print(tree.get_potential_valid([1]))
    print(tree.get_potential_valid([1, 3, 2]))