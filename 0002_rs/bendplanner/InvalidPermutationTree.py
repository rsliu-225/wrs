from treelib import Tree, Node


class IPTree:
    def __init__(self, num, name="invalid_tree"):
        self.node_ids = list(range(num))

        self.name = name
        self.tree = Tree(identifier=self.name)
        self.root = Node(tag="root", identifier="root", data=-1)
        self.tree.add_node(self.root)

    def show(self):
        self.tree.show()

    def add_invalid_seq(self, seq=None):
        if not seq:
            return

        str_seq = [str(i) for i in seq]
        for idx in list(range(len(seq))):
            if seq[idx] > self.node_ids[-1]:
                raise Exception(f"node {seq[idx]} exceeds the tree limit")

            node_id = "-" + '-'.join(str_seq[:idx + 1])
            if not self.tree.get_node(node_id):
                parent_node_id = node_id[:-len(str_seq[idx]) - 1]
                parent_node_id = parent_node_id if parent_node_id else "root"
                self.tree.create_node(
                    seq[idx], node_id, parent_node_id, seq[idx],
                )

    def whether_valid(self, seq=None):
        if not seq:
            return False

        node_id = ""
        cur = None

        for idx in seq:
            node_id += '-' + str(idx)
            cur = self.tree.get_node(node_id)
            if not cur:
                return True
            if cur.is_leaf():
                return False

        if not cur:
            return False
        else:
            remained_valid_children = len(self.node_ids) - len(seq) - \
                                      len(cur.successors(self.name))

            if remained_valid_children > 0:
                return True
            else:
                return False

    def get_potential_valid(self, pre_seq=None):
        # Use DFS to get the first potential valid
        available_child_ids = self.node_ids[::-1]
        pre_node_ids = ""
        if pre_seq:
            if self.whether_valid(pre_seq):
                for idx in pre_seq:
                    available_child_ids.remove(idx)
                    pre_node_ids += '-' + str(idx)
            else:
                print("Invalid sequence!")
                return []

        stack = [(pre_node_ids + '-' + str(idx), [x for x in available_child_ids if x != idx])
                 for idx in available_child_ids]

        while stack:
            # print(stack)
            node_id, child_ids = stack.pop()
            # print(node_id, child_ids)
            seq_invalid = self.tree.get_node(node_id) and self.tree.get_node(node_id).is_leaf()
            if not seq_invalid:
                if not child_ids:
                    return [int(x) for x in node_id.split('-')[1:]]
                for idx in child_ids:
                    stack.append((node_id + '-' + str(idx), [x for x in child_ids if x != idx]))

        print("Can not find any potentially valid sequence!")
        return []


if __name__ == '__main__':
    tree = IPTree(15)
    tree.add_invalid_seq([0])
    tree.add_invalid_seq([1, 2, 0])
    tree.add_invalid_seq([1, 3, 4])
    tree.add_invalid_seq([1, 2, 3])
    tree.add_invalid_seq([1, 2, 4])
    tree.tree.show()
    print(tree.get_potential_valid())

    print(tree.get_potential_valid([1]))
