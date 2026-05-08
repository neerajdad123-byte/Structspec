"""
Generate a PERFECT corpus matching your 20 prompts exactly.
Each prompt gets a high-quality Python implementation tokenized with the Qwen model.
"""

CORPUS_CODE = {
    "implement linked lsit in python only code no comments": """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def create_linked_list(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    current = head
    for val in arr[1:]:
        current.next = ListNode(val)
        current = current.next
    return head
def print_linked_list(head):
    current = head
    while current:
        print(current.val)
        current = current.next
""",
    "implement fibbanoci in python, no comments, only code:": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
def fib_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
""",
    "implement BST in python ,only code,no comments ": """
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None
def insert(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    return root
def search(root, val):
    if not root or root.val == val:
        return root
    return search(root.left, val) if val < root.val else search(root.right, val)
""",
    "reverse a list in python,only code,no comments": """
def reverse_list(arr):
    left, right = 0, len(arr)-1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr
def reverse_slice(arr):
    return arr[::-1]
""",
    "reverse linked list in python ,no coemmnts ,only code": """
def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev
""",
    "write function in python to check given number primr or not only code no comments": """
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i+2) == 0:
            return False
        i += 6
    return True
""",
    "implement merge sort in python only code no comments": """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""",
    "implement quick sort in python only code no comments": """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
""",
    "write fun to check given string palindrome or not in python only code no comments": """
def is_palindrome(s):
    left, right = 0, len(s)-1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
def is_palindrome_slice(s):
    return s == s[::-1]
""",
    "implement deletion at end in linked list in python only code no comments": """
def delete_at_end(head):
    if not head or not head.next:
        return None
    current = head
    while current.next and current.next.next:
        current = current.next
    current.next = None
    return head
""",
    "impleemnt insertion at begining in linked list in python only code no comments": """
def insert_at_beginning(head, val):
    new_node = ListNode(val)
    new_node.next = head
    return new_node
""",
    "implement deletion of node at begining in linked list in python only code no comments": """
def delete_at_beginning(head):
    if not head:
        return None
    return head.next
""",
    "implement Min heap in python only code no comments": """
class MinHeap:
    def __init__(self):
        self.heap = []
    def parent(self, i):
        return (i-1)//2
    def insert(self, val):
        self.heap.append(val)
        self.heapify_up(len(self.heap)-1)
    def heapify_up(self, i):
        while i > 0 and self.heap[self.parent(i)] > self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)
    def extract_min(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify_down(0)
        return min_val
    def heapify_down(self, i):
        smallest = i
        left = 2*i + 1
        right = 2*i + 2
        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right
        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self.heapify_down(smallest)
""",
    "implement max heap in python only code no comments": """
class MaxHeap:
    def __init__(self):
        self.heap = []
    def parent(self, i):
        return (i-1)//2
    def insert(self, val):
        self.heap.append(val)
        self.heapify_up(len(self.heap)-1)
    def heapify_up(self, i):
        while i > 0 and self.heap[self.parent(i)] < self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)
    def extract_max(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify_down(0)
        return max_val
    def heapify_down(self, i):
        largest = i
        left = 2*i + 1
        right = 2*i + 2
        if left < len(self.heap) and self.heap[left] > self.heap[largest]:
            largest = left
        if right < len(self.heap) and self.heap[right] > self.heap[largest]:
            largest = right
        if largest != i:
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            self.heapify_down(largest)
""",
    "implement Binart Tree in python only code no comments": """
class BinaryTreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None
def insert_bt(root, val):
    if not root:
        return BinaryTreeNode(val)
    if val < root.val:
        root.left = insert_bt(root.left, val)
    else:
        root.right = insert_bt(root.right, val)
    return root
""",
    "implement BST in python only code no comments": """
def bst_insert(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = bst_insert(root.left, val)
    else:
        root.right = bst_insert(root.right, val)
    return root
def bst_search(root, val):
    if not root or root.val == val:
        return root
    return bst_search(root.left, val) if val < root.val else bst_search(root.right, val)
""",
    "write function for reversing linked list in python only code no comments": """
def reverse_list_ll(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
""",
    "write function for checking wheather given number is prime or not only code no comments": """
def check_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True
""",
    "implement stack using list in python only code no comments": """
class Stack:
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("pop from empty stack")
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("peek from empty stack")
    def is_empty(self):
        return len(self.items) == 0
    def size(self):
        return len(self.items)
""",
    "implement queue using list in python only code no comments": """
class Queue:
    def __init__(self):
        self.items = []
    def enqueue(self, item):
        self.items.append(item)
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        raise IndexError("dequeue from empty queue")
    def is_empty(self):
        return len(self.items) == 0
    def size(self):
        return len(self.items)
""",
}

print(f"Corpus has {len(CORPUS_CODE)} entries")

# Now generate tokenized corpus using model
from llama_cpp import Llama
model_path = r"C:\Users\neera\.lmstudio\models\Qwen\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
llm = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=2048, logits_all=False, verbose=False)

corpus_out = {}
for name, code in CORPUS_CODE.items():
    text = "\n```python\n" + code.strip() + "\n```"
    ids = list(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
    tokens = []
    for tid in ids:
        piece = llm.detokenize([tid]).decode("utf-8", errors="ignore")
        tokens.append({"id": tid, "token": piece})
    corpus_out[name] = {"name": name, "tokens": tokens, "code": code}

out_path = r"C:\Users\neera\OneDrive\Desktop\structspec\Structspec\matched_dsa_corpus.json"
import json
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(corpus_out, f, indent=2)
print(f"\nWrote matched corpus to {out_path}")
print(f"Total examples: {len(corpus_out)}")
