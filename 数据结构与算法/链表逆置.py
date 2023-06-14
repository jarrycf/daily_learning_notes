def reverse_linked_list(head): # 1, 2, 3
    prev = None # 用于保存当前节点的前一个节点
    current = head # 链表的第一个节点  1 # ListNode(1, ListNode(2, ListNode(3)))
    while current:
        next_node = current.next # 保存当前节点的下一个节点 2
        current.next = prev # 当前节点的next指向prev，即将当前节点反转 No 1变为None
        prev = current # 1
        current = next_node  # 2
    return prev