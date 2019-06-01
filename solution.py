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
    # leetcode1
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

    # leetcode2
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

    # leetcode3
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
        while right < len(s):
            if tmp[s[right]] == -1:
                tmp[s[right]] = right
                right += 1
                res = max(res, right - left)
            else:
                tmp[s[right]] = right
                while left < tmp[s[right]]:
                    tmp[s[left]] = -1
                    left += 1

        return res

    # leetcode5
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(s) < 1:
            return ''
        res = s[0]
        left = 0
        while left < len(s):
            right = len(s) - 1
            while right > left:
                if s[left] == s[right]:
                    if self.isPalindrome(s, left, right):
                        if len(s[left:right + 1]) > len(res):
                            res = s[left:right + 1]
                        break
                    else:
                        right -= 1
                else:
                    right -= 1

            left += 1
        return res

    def isPalindrome(self, s, left, right):
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return False
        return True

    # leetcode5,516
    def longestPalindrome1(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(s) < 2:
            return s
        left = 0
        right = 0
        maxlen = len(s)
        dp = [([0] * maxlen) for i in xrange(maxlen)]
        dp[0][0] = 1
        for i in xrange(1, maxlen):
            dp[i][i] = 1
            dp[i][i - 1] = 1

        for k in range(2, maxlen + 1):
            for i in range(maxlen - k + 1):
                if s[i] == s[i + k - 1] and dp[i + 1][i + k - 2] == 1:
                    dp[i][i + k - 1] = 1
                    if right - left + 1 < k:
                        left = i
                        right = i + k - 1

        # 5 return
        # return s[left:right + 1]
        # 516 return
        return right - left + 1

    # leetcode7
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        res = ''
        if x > 0:
            length = len(str(x))
        else:
            length = len(str(abs(x)))
        # print length
        x_tmp = abs(x)
        for i in range(length):
            tmp = x_tmp % 10
            # print tmp
            x_tmp = x_tmp / 10
            # print x
            res += str(tmp)

        if int(res) > pow(2, 31):
            return 0
        elif x > 0:
            return int(res)
        else:
            return -int(res)

    # leetcode9
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        res = ''
        length = len(str(x))
        if length < 2:
            return True
        x_tmp = abs(x)
        for i in range(length):
            tmp = x_tmp % 10
            x_tmp = x_tmp / 10
            res += str(tmp)

        if int(res) == x:
            return True

    # leecode13
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        romanInt = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        num = []
        res = 0
        for i in range(len(s)):
            num.append(romanInt[s[i]])

        for i in range(len(s) - 1):
            if num[i] < num[i + 1]:
                res -= num[i]
            else:
                res += num[i]

        return res + num[len(s) - 1]

    # leetcode15
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        nums = sorted(nums)
        print nums
        for i in range(len(nums)):
            start = i + 1
            end = len(nums) - 1
            if start >= end:
                break
            else:
                while start < end:
                    if nums[i] + nums[start] + nums[end] > 0:
                        end -= 1
                    elif nums[i] + nums[start] + nums[end] < 0:
                        start += 1
                    else:
                        if [nums[i], nums[start], nums[end]] not in res:
                            res.append([nums[i], nums[start], nums[end]])
                        end -= 1
                        start += 1

        return res

    # leetcode14
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
            return ""

        res = strs[0]
        print res
        for i in range(1, len(strs)):
            while not strs[i].startswith(res):
                print res
                res = res[0:len(res) - 1]
        return res

    # leetcode16
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        if len(nums) < 3:
            return 0
        tmp = abs(target - (nums[0] + nums[1] + nums[2]))
        res = nums[0] + nums[1] + nums[2]
        nums = sorted(nums)
        for i in range(len(nums)):
            start = i + 1
            end = len(nums) - 1
            while start < end:
                s = nums[i] + nums[start] + nums[end]
                if s == target:
                    return target
                elif s < target:
                    if target - s < tmp:
                        tmp = target - s
                        res = s
                    start += 1
                else:
                    if s - target < tmp:
                        tmp = s - target
                        res = s
                    end -= 1
        return res

    # leetcode12
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        romans = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        res = ''
        for i in range(len(values)):
            while num >= values[i]:
                res += romans[i]
                num -= values[i]
        return res

    # importantleetcode17
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        dfs实践
        """

        def _dfs(num, string, res):
            if num == length:
                res.append(string)
                return
            if digits[num] == '0' or digits[num] == '1':
                _dfs(num + 1, string, res)
            for i in range(len(dicts[digits[num]])):
                _dfs(num + 1, string + dicts[digits[num]][i], res)

        res = []
        dicts = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz',
                 '1': '', '0': ''}
        length = len(digits)
        _dfs(0, '', res)
        if res == [""]:
            return []
        return res

    # leetcode18
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        if len(nums) < 3:
            return
        res = []
        nums = sorted(nums)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                start = j + 1
                end = len(nums) - 1
                while start < end:
                    if nums[i] + nums[j] + nums[start] + nums[end] == target:
                        if [nums[i], nums[j], nums[start], nums[end]] not in res:
                            res.append([nums[i], nums[j], nums[start], nums[end]])
                        start += 1
                        end -= 1
                    elif nums[i] + nums[j] + nums[start] + nums[end] < target:
                        start += 1
                    else:
                        end -= 1

        return res

    # leetcode20
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        data = {'(': ')', '[': ']', '{': '}'}
        keys = []
        for i in range(len(s)):
            if s[i] in data.keys():
                keys.append(s[i])
            else:
                try:
                    if s[i] == data[keys[-1]]:
                        keys.pop(-1)
                    else:
                        return False
                except:
                    return False
        if len(keys) == 0:
            return True
        else:
            return False

    # leetcode22
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """

        def _dfs(left, right, out, res):
            if left < 0 or right < 0 or left > right:
                return
            if left == 0 and right == 0:
                res.append(out)
                return
            _dfs(left - 1, right, out + '(', res)
            _dfs(left, right - 1, out + ')', res)

        res = []
        _dfs(n, n, '', res)
        return res


if __name__ == '__main__':
    print Solution().generateParenthesis(3)
