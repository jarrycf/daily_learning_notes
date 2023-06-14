def merge_sorted_lists(l1, l2): # l1: 1, 2, 3, 4   l2: 2, 4, 6, 8
    dummy = ListNode() # 一个虚拟节点 dummy，用于存储合并后的链表。
    current = dummy
    while l1 and l2: # 直到 l1 或 l2 中的任意一个链表为空
        if l1.val < l2.val: # 1<2
            current.next = l1 # 1, 2, 2, 3, 4, 4, 6, 8
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next 
    current.next = l1 if l1 else l2 # 将非空链表的剩余部分直接连接到合并链表的末尾，表示合并操作完成。
    return dummy.next # 返回虚拟节点 dummy 的下一个节点，即合并后的链表的头节点