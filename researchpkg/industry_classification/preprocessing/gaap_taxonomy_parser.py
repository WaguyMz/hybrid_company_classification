import os
import sys
from enum import Enum
from functools import lru_cache
from typing import Dict

dir_path = os.path.realpath(os.path.join(__file__, "..", "..", ".."))
sys.path.append(dir_path)
import treelib
import xbrl.taxonomy as tx
from xbrl.cache import HttpCache


def get_concepts_dict(root_tax_dir, tax_name, tax_version):
    """
    Load the concepts dictionary from the taxonomy.
    :param root_tax_dir  The root directory of the taxonomies
    :param tax_name: The name of the taxonomy . e.g us-gaap
    :param tax_version: The version of the taxonomy. e.g 2023
    """
    cache = HttpCache("/tmp/xbrl_cache")
    tax_full_name = f"{tax_name}-{tax_version}"
    tax_dir = os.path.join(root_tax_dir, tax_full_name)
    concepts_file = os.path.join(tax_dir, "elts", f"{tax_full_name}.xsd")
    concepts_dict = tx.parse_taxonomy(concepts_file, cache).concepts
    return concepts_dict


class CalculationTreeType(Enum):
    """
    An enumeration of the types of the taxonomy trees.
    """

    BALANCE_SHEET = 1
    INCOME_STATEMENT = 2
    CASH_FLOW_STATEMENT = 3
    INCOME_STATEMENT_V2 = 4


class CalculationTreeNode:
    def __init__(self, concept_id, parent_concept_id, concept_name, weight=1, level=1):
        self.concept_id = concept_id
        self.parent_concept_id = parent_concept_id
        self.concept_name = concept_name
        self.weight = weight
        self.level = level
        self.children = []
        self.dfs_stack = []
        self.number = 0

    def numerotate(self, current_number=[0]):
        """
        Recursively add a number to the node and its children.
        """
        self.number = current_number[0]
        for child in self.children:
            current_number[0] += 1
            child.numerotate(current_number)

    def build_dfs_stack(self):
        dfs_stack = []
        for child in self.children:
            dfs_stack.extend(child.build_dfs_stack())
        dfs_stack.append(self)  # Add the current node at the end
        return dfs_stack

    @staticmethod
    def merge_trees(trees: list):
        """
        Merge a list of trees into a single tree.
        By adding a new root node and making the original trees its children.
        """

        root_node = CalculationTreeNode("root", None, "root", weight=1, level=0)
        root_node.children = trees
        root_node.dfs_stack = root_node.build_dfs_stack()
        return root_node

    @staticmethod
    def build_from_locator(
        root_locator, concepts_dict, weight, level=1, parent_concept_id=None
    ):
        concept_id = root_locator.concept_id
        concept_name = concepts_dict[concept_id].name
        node = CalculationTreeNode(
            concept_id, parent_concept_id, concept_name, weight, level
        )
        node.level = level
        for child_arc in root_locator.children:
            child_locator = child_arc.to_locator
            node.children.append(
                CalculationTreeNode.build_from_locator(
                    child_locator,
                    concepts_dict,
                    weight=child_arc.weight,
                    level=level + 1,
                    parent_concept_id=concept_id,
                )
            )
        return node

    def display(self):
        indent = "  " * self.level
        print(f"{indent}{self.concept_name}({self.level}, {self.number  }) (Weight: {self.weight})")
        for child in self.children:
            child.display()

    def get_node_by_concept_id(self, concept_id):
        if self.concept_id == concept_id:
            return self
        for child in self.children:
            result = child.get_node_by_concept_id(concept_id)
            if result:
                return result
        return None

    def get_node_by_concept_name(self, concept_name):
        if self.concept_name == concept_name:
            return self
        for child in self.children:
            result = child.get_node_by_concept_name(concept_name)
            if result:
                return result

        return None

    def size(self):
        size = 1
        for child in self.children:
            size += child.size()
        return size

    def __iter__(self):
        """
        Define an iterator for the tree nodes using Breadth-First Search (BFS).
        """
        return self.bfs_iterator()

    def bfs_iterator(self):
        """
        Breadth-First Search (BFS) iterator for the tree nodes.
        """
        queue = [self]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.children)

    def dfs_iterator(self):
        """
        Reverse Breadth-First Search (BFS) iterator for the tree nodes using a pre-built stack.
        """
        for node in self.dfs_stack:
            if node.concept_id != "root":
                yield node

    def __str__(self):
        return f"({self.concept_name}, {self.weight})"

    def __repr__(self):
        return f"({self.concept_name}, {self.weight})"


class CalculationTree:
    def __init__(self, type=CalculationTreeType.BALANCE_SHEET):
        self.root_trees = []
        self.root: CalculationTreeNode = None  # type: ignore
        self.type = type

    def size(self):
        size = 0
        for root_tree in self.root_trees:
            size += root_tree.size()
        return size

    @staticmethod
    def build_taxonomy_tree(
        root_tax_dir, tax_name, tax_version, type=CalculationTreeType.BALANCE_SHEET
    ):
        """
        Build the taxonomy tree from the taxonomy.
        :param root_tax_dir  The root directory of the taxonomies
        :param tax_name: The name of the taxonomy . e.g us-gaap
        :param tax_version: The version of the taxonomy. e.g 2023
        :param concepts_dict: The concepts dictionary of the taxonomy
        :param type : The type of the calculation tree: \
        CalculationTreeType.BALANCE_SHEET,
         CalculationTreeType.INCOME_STATEMENT,
         CalculationTreeType.CASH_FLOW_STATEMENT

        """

        # 1.Load the concepts dictionary from the taxonomy
        max_retries = 5
        nb_attempts = 0
        concepts_dict = None
        while nb_attempts < max_retries:
            try:
                concepts_dict = get_concepts_dict(root_tax_dir, tax_name, tax_version)
                break
            except Exception as e:
                nb_attempts += 1
                print(f"Error loading concepts dictionary: {e}")
        
        if concepts_dict is None:
            raise Exception(f"Failed to load concepts dictionary after {max_retries} attempts")                
        

        # 2.Load the calculation linkbase from the taxonomy
        cache = HttpCache("/tmp/xbrl_cache")
        tax_full_name = f"{tax_name}-{tax_version}"
        tax_dir = os.path.join(root_tax_dir, tax_full_name)

        if type == CalculationTreeType.BALANCE_SHEET:
            tax_calc_file = os.path.join(
                tax_dir, f"stm/us-gaap-stm-sfp-cls-{tax_version}.xsd"
            )
        elif type == CalculationTreeType.INCOME_STATEMENT:
            tax_calc_file = os.path.join(
                tax_dir, f"stm/us-gaap-stm-soi-{tax_version}.xsd"
            )
        
        elif type == CalculationTreeType.INCOME_STATEMENT_V2:
            tax_calc_file = os.path.join(
                tax_dir, f"stm/us-gaap-stm-soi-{tax_version}.xsd"
            )
        else:
            raise NotImplemented(f"Unsupported calculation tree type: {type}")

        taxonomy = tx.parse_taxonomy(tax_calc_file, cache)
        calc_linkbase = CalculationTree()
        root_locators = taxonomy.cal_linkbases[0].extended_links[0].root_locators
        for root_locator in root_locators:
            calc_linkbase.root_trees.append(
                CalculationTreeNode.build_from_locator(root_locator, concepts_dict, 1)
            )

        calc_linkbase.root = CalculationTreeNode.merge_trees(calc_linkbase.root_trees)
        calc_linkbase.root.numerotate()

        return calc_linkbase

    def display(self):
        for i, root_tree in enumerate(self.root_trees):
            print(f"Displaying Tree {i + 1}:")
            root_tree.display()
            print("\n")

    @lru_cache(maxsize=10000)
    def get_node_by_concept_id(self, concept_id) -> CalculationTreeNode:
        for root_tree in self.root_trees:
            node = root_tree.get_node_by_concept_id(concept_id)
            if node:
                return node

        return None

    @lru_cache(maxsize=10000)
    def get_node_by_concept_name(self, concept_name) -> CalculationTreeNode:
        for root_tree in self.root_trees:
            node = root_tree.get_node_by_concept_name(concept_name)
            if node:
                return node

        return None

    def __getitem__(self, concept_name):
        return self.get_node_by_concept_name(concept_name)

    def apply(self, input_datas, min_level=0, max_level=10000) -> Dict:
        """
        Apply the calculation linkbase on the input data.
        :param input_datas: A dictionary containing the input data. key: concept_name, value: concept_value
        :return: A dictionary containing the output data. key: concept_name, value: concept_value
        """
        # Create an empty dictionary to store the output values
        calculated_values_dict = {}
        output_dict = {}

        # Perform a reverse Breadth-First Search (BFS) of the calculation tree
        for node in self.root.dfs_iterator():
            concept_name = node.concept_name

            # If the concept is in the input_data, put it in the calculated_values_dict with its value
            if concept_name in input_datas:
                calculated_values_dict[concept_name] = input_datas[concept_name]
            else:
                # If it's a leaf node (no children), put it in the calculated_values_dict with a value of 0
                if not node.children:
                    calculated_values_dict[concept_name] = 0
                else:
                    # Iterate over its children to calculate its value
                    value = 0
                    for child in node.children:
                        child_concept_name = child.concept_name
                        child_value = calculated_values_dict[child_concept_name]

                        # Calculate the contribution of the child to the parent node's value
                        contribution = child_value * child.weight
                        value += contribution

                    # Put the calculated value in the calculated_values_dict for the parent node
                    calculated_values_dict[concept_name] = value

            value = calculated_values_dict[concept_name]
            if value != 0 and min_level <= node.level <= max_level:
                output_dict[concept_name] = calculated_values_dict[concept_name]

        return output_dict

    def normalize_by_parent(self, input_datas):
        """
        Apply the calculation linkbase on the input data. and normalize(divide) the output by the parent tag if exists
        :param input_datas: A dictionary containing the input data. key: concept_name, value: concept_value. At this stage
        the input datas as be completed (No zero node between a non zero node and the root) and contains only taxonomy tags.

        :return: A dictionary containing the output data. key: concept_name, value: concept_value
        """
        # Create an empty dictionary to store the output values

        output_dict = {}
        # Perform a reverse Breadth-First Search (BFS) of the calculation tree
        for node in self.root.dfs_iterator():
            concept_name = node.concept_name

            if concept_name not in input_datas:
                value = 0
            else:
                value = input_datas[concept_name]
                parent_node = node.parent_concept_id

                if parent_node:
                    parent_concept_name = self.get_node_by_concept_id(
                        parent_node
                    ).concept_name
                    value = value / input_datas[parent_concept_name]

            output_dict[concept_name] = value

        return output_dict

    def get_all_tags(self, max_level=1e5):
        """
        Return a list of all the tags in the calculation tree.
        """
        tags = []
        for node in self.root.dfs_iterator():
            if node.level <= max_level:
                tags.append(node.concept_name)
        return tags

    @lru_cache(maxsize=10000)
    def is_ancestor(
        self, node1: CalculationTreeNode, node2: CalculationTreeNode
    ) -> bool:
        """
        Check if node1 is an ancestor of node2.
        """
        if node1 == node2:
            return True
        while node2.parent_concept_id:
            node2 = self.get_node_by_concept_id(node2.parent_concept_id)
            if node2 == node1:
                return True
        return False

    def are_in_same_branch(self, concept_name1: str, concept_name2: str) -> bool:
        """
        Return true if concept_name1 and concept_name2 are in the same branch of the calculation tree.
        """

        node1 = self.get_node_by_concept_name(concept_name1)
        node2 = self.get_node_by_concept_name(concept_name2)

        if node1.level > node2.level:
            node1, node2 = node2, node1

        return self.is_ancestor(node1, node2)

    def depth_count_dict(self) -> dict:
        """
        Return the number of tags per depth in the calculation tree.
        """
        tags_per_depth = {}
        for node in self.root.dfs_iterator():
            if node.level not in tags_per_depth:
                tags_per_depth[node.level] = 0
            tags_per_depth[node.level] += 1
        return tags_per_depth
