class Question :
    def __init__(self, id, question, left_answer, right_answer) :
        self.id = id
        self.question = question
        self.left_answer = left_answer
        self.right_answer = right_answer
        self.left = None
        self.right = None
        self.parent = None


class QuestionTree :
    def __init__(self) :
        self.root = None

    def find_question(self, id) :
        def helper(node) :
            if not node : return None
            if node.id == id : return node
            return helper(node.left) or helper(node.right)   

        return helper(self.root)  

    def add_question(self, new_question, parent_id=None, direction="left"):
        if self.root is None:
            self.root = new_question
            print(f"Added root question with id {new_question.id}")
            return

        parent_node = self.find_question(parent_id)
        if not parent_node:
            print(f"Parent question with id {parent_id} not found.")
            return

        if direction == "left":
            if parent_node.left is None:
                parent_node.left = new_question
                new_question.parent = parent_node
                print(f"Added question with id {new_question.id} to the left of parent {parent_id}")
            else:
                print(f"Left child already exists for parent {parent_id}.")
        elif direction == "right":
            if parent_node.right is None:
                parent_node.right = new_question
                new_question.parent = parent_node
                print(f"Added question with id {new_question.id} to the right of parent {parent_id}")
            else:
                print(f"Right child already exists for parent {parent_id}.")
        else:
            print("Invalid direction. Use 'left' or 'right'.")


    def delete_question(self, id):
        node_to_delete = self.find_question(id)
        if not node_to_delete:
            print(f"Question with id {id} not found.")
            return

        def recursive_delete(node):
            if node.left:
                recursive_delete(node.left)
            if node.right:
                recursive_delete(node.right)
            node.left = None
            node.right = None

        if node_to_delete.parent:
            if node_to_delete.parent.left == node_to_delete:
                node_to_delete.parent.left = None
            else:
                node_to_delete.parent.right = None

        recursive_delete(node_to_delete)
        del node_to_delete
        print(f"Question with id {id} deleted.")
    
    def build_tree_from_file(self, filename):
        nodes = {}
        with open(filename, 'r') as file:
            for line in file:
                id, question, left_answer, right_answer = line.strip().split('|')
                node = Question(int(id), question, left_answer, right_answer)
                nodes[int(id)] = node
                if self.root is None:
                    self.root = node
        
        for id, node in nodes.items():
            if int(id) * 2 in nodes: 
                node.left = nodes[int(id) * 2]
                nodes[int(id) * 2].parent = node
            if int(id) * 2 + 1 in nodes:
                node.right = nodes[int(id) * 2 + 1]
                nodes[int(id) * 2 + 1].parent = node


        
    