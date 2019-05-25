#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: v1.0
@author: caiminchao
@corp: caiminchao@corp.netease.com
@software: PyCharm，python 2.7
@file: test
@time: 2019/5/23 20:51
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    # leetcode 1
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        res = {}
        for i in xrange(len(nums)):
            if target - nums[i] in res:
                return [res[target - nums[i]], i]
            res[nums[i]] = i

    # leetcode 2
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        l1 = ListNode(2)
        l1.next = ListNode(4)
        l1.next.next = ListNode(3)
        l2=ListNode(5)
        l2.next = ListNode(6)
        l2.next.next = ListNode(4)
        print Solution().addTwoNumbers(l1,l2).val
        """
        carry = 0
        n = ListNode(0)
        root = n
        while l1 or l2 or carry:
            v1 = 0
            v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1 + v2 + carry, 10)
            n.next = ListNode(val)
            n = n.next
        return root.next

    # 3
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        维护一个滑动窗口，维护数据结构为Hashmap
        """
        left = 0
        right = 0
        res = 0
        tmp = {}
        for i in xrange(len(s)):
            tmp[s[i]] = -1
        while right<len(s):
            if tmp[s[right]] == -1:
                tmp[s[right]] = right
                right += 1
                res = max(res,right-left)
            else:
                tmp[s[right]] = right
                while left<tmp[s[right]]:
                    tmp[s[left]] = -1
                    left+=1

        return res
    # 5
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """


if __name__ == '__main__':
    print Solution().lengthOfLongestSubstring('pwwketmp[s[right]] = rightw')
    print Solution().lengthOfLongestSubstring('abcabcbb')
    print Solution().lengthOfLongestSubstring('abba')
