def remove_duplicate(head):
    if not head or not head.next: # 检查链表是否为空或只有一个节点
        return head # 是的话 直接返回原链表头节点
    current = head # 初始化为链表的头节点
    while current:
        runner = current # 初始化为当前节点
        while runner.next: # 遍历 runner 之后的节点
            if runner.next.val == current.val: # 找到重复节点
                runner.next = runner.next.next # 将 runner 的下一个节点指向下一个的下一个节点，即将重复节点从链表中删除。
            else: # 如果未找到重复节点，将 runner 移动到下一个节点
                runner = runner.next 
        current = current.next # 更新 current 为下一个节点
    return head


