##  呼延伟同学整理的分类题库

> 每道题写出思路，暴力解-->最优解 ，时间、空间复杂度



### 动态规划

> **动态规划三要素**：明确状态与选择，明确dp数组的含义，根据选择写出状态转移方程，
>
> 空间优化方法：一维变常量，二维变一维
>
> **python基础知识**：Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况



#### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/) 

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # step1 定义并初始化dp数组
        m = len(word1) # 行
        n = len(word2) # 列
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        # dp[i][j] 表示word1[0,..i] 转到 word2[0,...j] 所使用的最小操作数
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        
        # step2 
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] # 字符相等，不用操作
                else:
                    dp[i][j] = min(dp[i-1][j], # 表示删除操作
                                   dp[i][j-1], # 表示插入操作
                                   dp[i-1][j-1]) # 表示替换操作
                                    + 1
        return dp[m][n]
```

#### [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 定义并初始化dp
        # dp[i] 表示以nums[i] 为结尾的最大连续子数组和
        dp = nums
        for i in range(1, len(nums)):
            dp[i] = max(nums[i], dp[i-1] + nums[i])
        return max(dp)
```



#### [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/) 

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        #　动态规划
        n = len(nums)
        maxDp = [1 for _ in range(n+1)]
        minDp = [1 for _ in range(n+1)]
        ans = float('-inf')
        for i in range(1, n+1):
            maxDp[i] = max(maxDp[i-1]*nums[i-1],minDp[i-1]*nums[i-1],nums[i-1])
            minDp[i] = min(maxDp[i-1]*nums[i-1], minDp[i-1]*nums[i-1], nums[i-1])
            ans = max(ans, maxDp[i])
        return ans
```



#### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # 思路1 定义并初始化dpO(n2),O(n)
        # dp[i] 表示以nums[i] 为结尾的最长地政子序列的长度
        n = len(nums)
        dp = [1 for _ in range(n)]
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j]+1) 
        return max(dp)

        # 思路2 贪心+二分查找，能求出最长递增子序列是谁
        res = []
        for one in nums:
            if not res or res[-1] < one:
                res.append(one)
            else: # 找到要插入的位置，如果这个位置有其他元素，直接替换
                l, r = 0, len(res) - 1
                locate = r
                while l <= r:
                    mid = (l + r) // 2
                    if res[mid] >= one:
                        locate = mid
                        r = mid -1
                    elif res[mid] < one:
                        l = mid + 1
                res[locate] = one 
        return len(res)

```



#### [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # 定义并初始化
        # dp[i][j] 表示text1[0:i]和text2[0:j] 的最长公共子序列长度
        m = len(text1) # 列数
        n = len(text2) # 行数
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(1,m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

# 求最长公共子序列是谁
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # 定义并初始化
        # dp[i][j] 表示text1[0:i]和text2[0:j] 的最长公共子序列长度
        m = len(text1) # 列数
        n = len(text2) # 行数
        dp = [['' for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(1,m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + text1[i-1]
                else:
                  	dp[i][j] = dp[i-1][j] if len(dp[i-1][j]) > len(dp[i][j-1]) else dp[i][j-1]
                    # dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
```



#### [718. 最长重复子数组（最长公共子串）](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

```python
# 问题：最长公共字串的长度
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        # dp[i][j] 表示 以字符串s1[0...i] 和 s2[0...j] 的最长公共字串的长度
        m = len(A)
        n = len(B)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        res = 0
        for i in range(1, m+1):
            for j in range(1, n+1):
                if A[i-1] == B[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    res = max(res, dp[i][j])
        return res
```



#### [583. 两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)

```python
# 问题：每次任意删除一个字母，使得两个单词相同的最小次数
class Solution:
    def lca(self, word1, word2):
        m = len(word1)
        n = len(word2)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    def minDistance(self, word1: str, word2: str) -> int:
        # 思路：word1 删除到最长公共子序列，word2 删除到最长公共子序列
        return len(word1) + len(word2) - 2*self.lca(word1, word2)


```



#### [712. 两个字符串的最小ASCII删除和](https://leetcode-cn.com/problems/minimum-ascii-delete-sum-for-two-strings/)

```python
# 问题：删除的字母的ascii 最小
# 知识点：print(chr(65)), print(ord('A'))
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        # 定义并初始化dp
        # dp[i][j] 表示s1[i:] 和 s2[j:] 达到相等所需要产出最少ascii的最小和，最总答案为dp[0][0]
        m = len(s1)
        n = len(s2)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        for i in range(m-1, -1, -1):
            dp[i][n] = dp[i+1][n] + ord(s1[i])
        
        for j in range(n-1, -1, -1):
            dp[m][j] = dp[m][j+1] + ord(s2[j])

        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                if s1[i] == s2[j]:
                    dp[i][j] = dp[i+1][j+1] # 相等说明不用删， 注意赋值用= 号
                else:
                    dp[i][j] = min(dp[i+1][j] + ord(s1[i]), dp[i][j+1] + ord(s2[j]))
        return dp[0][0]
```



#### [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        # 定义并初始化:dp[i][j] 表示，s 的第 i 个字符到第 j 个字符组成的子串中 最长回文子序列的长度
        # base case dp[i][i] = 1
        n = len(s)
        dp = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
            for j in range(i-1, -1, -1):
                if s[j] == s[i]:
                    dp[j][i] = dp[j+1][i-1] + 2 # 注意：是dp[j][i], 不是dp[i][j]
                else:
                    dp[j][i] = max(dp[j+1][i], dp[j][i-1])
        return dp[0][n-1]
```



#### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

   ```python
   #思路1：中心扩展法，分奇数偶数 O(n^2)， O(1)
   class Solution:
       def helper(self, s, l, r):
           while l >=0 and r < len(s) and s[l] == s[r]:
               l -= 1
               r += 1
           return s[l+1:r] # 注意边界点
       def longestPalindrome(self, s: str) -> str:
           res = ''
           for i in range(len(s)):
               one = self.helper(s, i, i)
               two = self.helper(s, i, i+1)
               if len(one) > len(res):
                   res = one
               if len(two) > len(res):
                   res = two
           return res
   # 思路2：dp
   ```

   

#### [1312. 让字符串成为回文串的最少插入次数](https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)

```python
# 思路1：插入次数，可以转变成删除次数，然后求s和s[::-1]的最长公共子序列
class Solution:
    def lcs(self, text1, text2):
        m = len(text1) # 列数
        n = len(text2) # 行数
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(1,m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
    
    def minInsertions(self, s: str) -> int:
        l = self.lcs(s,s[::-1])
        return len(s) - l
```



#### [132. 分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)

```python
# 问题：使一个字符串每个部分都是回文串的最小切分次数
class Solution:
    def minCut(self, s: str) -> int:
        # dp[i] 表示范围 s[0:i], 不包括s[i]，最少要分割的次数为dp[i]
        # 设 j 是 0 ~ i 的一个切分点； 如果 j ~i 是回文串，则 dp[i] = min(dp[i], dp[j] + 1)
        # base case dp[0] = -1 ,dp[1]表示s[0:1] 为回文最少切割次数，显然为0，  最总返回dp[n]

        n = len(s)
        dp = [i for i in range(-1, n)] # 注意初始化
        for i in range(1, n+1):
            for j in range(i):
                if s[j:i] == s[j:i][::-1]:
                    dp[i] = min(dp[i], dp[j] + 1)
        return dp[n]

```



#### [647. 回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)

```python
# 问题：给定一个字符串，你的任务是计算这个字符串中有多少个回文子串
```

#### [409. 最长回文串](https://leetcode-cn.com/problems/longest-palindrome/)

```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        temp = {}
        for char in s:
            temp[char] = temp.setdefault(char, 0) + 1  
        res = 0
        flag = 0
        for key, val in temp.items():
            if val % 2 == 0:
                res += val
            else:
                res += val - 1 # 重点 ，如果字母是奇数个，选择val - 1个
                flag = 1
        return res + flag
```



#### [10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)

```python
# 问题：字符串s和规律串p 能否匹配上
# '.' 匹配任意单个字符
# '*' 匹配零个或多个前面的那一个元素
# 思路1：递归
# for循环写法
def isMatch1(text, pattern):
    if len(text) != len(pattern):
        return False
    for j in range(len(pattern)):
        if pattern[j] != text[j]:
            return False
    return True

# 递归写法
def isMatch2(text, pattern):
    if not pattern:
        return text == ''
    fistMatch = text != '' and pattern[0] == text[0] # bool 类型
    return fistMatch and isMatch2(text[1:], pattern[1:])

# 处理. ：可以匹配任意一个字符
def isMatch3(text, pattern):
    if not pattern:
        return text == ''
    fistMatch = text != '' and pattern[0] in [text[0], '.']
    return fistMatch and isMatch3(text[1:], pattern[1:])

# 处理* ：可以匹配前面出现字符的任意次数，包括0次
def isMatch4(text, pattern):
    if not pattern:
        return text == ''
    fistMatch = text != '' and pattern[0] in [text[0], '.']
    # 如果 当前pattern的长度大于2，并且第2个字符是* （下标索引是1）
    if len(pattern) >= 2 and pattern[1] == '*':
        return isMatch4(text, pattern[2:]) or fistMatch and isMatch4(text[1:], pattern)
    else:
        return fistMatch and isMatch4(text[1:], pattern[1:])

# 思路2：dp
```

#### [44. 通配符匹配](https://leetcode-cn.com/problems/wildcard-matching/)

```python
# '？' 匹配任意单个字符
# '*' 匹配任意一个字符串
# 思路1：递归





# 思路2：动态规划
```



#### [887. 鸡蛋掉落](https://leetcode-cn.com/problems/super-egg-drop/)`困难`

#### [312. 戳气球](https://leetcode-cn.com/problems/burst-balloons/)`困难`

```python
# 问题：nums = [3,1,5,8] ，每次删除一个数字，会获得 
# nums[i - 1] * nums[i] * nums[i + 1] 枚硬币，边界视为1
# 求获得硬币的最大数量
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)
        nums.insert(0,1)
        nums.append(1)
        # 初始化并定义dp数组
        # 在原始数组的收尾分别插入一个1，数组变成了n+2
        # 原问题转化成：戳破气球0和气球n+1之间的全部气球，最多能获得多少分
        # dp[i][j] = x 表示：戳破i和j之间（不包括i和j）的所有气球所能获最多x分
        # 最终结果是dp[0][n+1], base case :当j <= i+1,dp[i][j] = 0
        # 初始化dp
        dp = [[0 for _ in range(n+2)] for _ in range(n+2)]
        # 0 <= i <= n+1; i+1 < j <= n+1 
        # i < k < j
        # 设k为i，j 之间最后戳破的那个气球
        # dp[i][j] = dp[i][k] + dp[k][j] + points[i]*points[k]*points[j]
        # 不好确定范围就画图
        for i in range(n, -1, -1):
            for j in range(i+1, n+2):
                for k in range(i+1, j):
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + nums[i]*nums[k]*nums[j])
        return dp[0][n+1]
```



#### [877. 石子游戏](https://leetcode-cn.com/problems/stone-game/) `中等`

```python
# 问题：偶数堆石子（石子总数是奇数），每次选取开始或结尾整堆石子，A先选，最后A是否能赢，返回true or false？例如 nums = [5,3,4,5]

```



#### [651. 4键键盘](https://leetcode-cn.com/problems/4-keys-keyboard/) `中等`

```python
# 问题：操作：打印A、ctrl+a， ctrl+c， ctrl+v ，操作N次数，屏幕字母最多
# 最优的按键顺序一定是下面两种情况：
# 1. 一直按A:A,A,A......
# 2. A,A,A....c+a,c+c,c+v,c+a,c+c,c+v,.....(c+a,c+c,c+v 循环下去)
# 状态：剩余的敲击次数n；选择：四种选择
# dp[i] 表示i次操作后，最多能显示多少个A

```

#### 股票简介

```shell
1. dp含义：
dp[3][2][1] 的含义：今天是第三天，我现在手上持有着股票，至今最多进行 2 次交易
dp[2][3][0] 的含义：今天是第二天，我现在手上没有持有股票，至今最多进行 3 次交易
最终返回结果：dp[n - 1][K][0]

```

![](https://mmbiz.qpic.cn/mmbiz_png/map09icNxZ4nPicwNq5syrSwnBc02yxG3aLFHicK3LhVZXEJvHzEOgGpjp8RzCxIkQpW0K7qGkqYKcCP5jdJIrpibA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



```shell
2. base case
```

![](https://mmbiz.qpic.cn/mmbiz_png/map09icNxZ4nPicwNq5syrSwnBc02yxG3akByqn8e7kyr0hSKS6iaVkicDsZrc08oic4wp5c7sPk7LzicGJm3xlBRSew/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



```shell
3. 总结
```

![](https://mmbiz.qpic.cn/mmbiz_png/map09icNxZ4nPicwNq5syrSwnBc02yxG3aewN24fa7UR8G7byHOb7lUfrlgCkUN1KsL5PYsIicKfE0mQ2OCibCXiajA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



#### [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/) 

```python
# 问题描述：交易1次（买卖各一次，算一笔交易，并且必须先买后卖）
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 思路1：贪心算法,
        # 时空：O(n)，O(1)
        low = float('inf') # 初始化最低股票，设为正无穷
        res = 0
        for i in range(len(prices)):
            low = min(low, prices[i]) # 选择当前最小的股票价格
            res = max(res, prices[i] - low) # 获取当前最大的股票收益
        return res

        # 思路2：dp
        # 时空：O(n)，O(n)
        # dp[i][0] 表示第i天 不持有 股票所得现金
        # dp[i][1] 表示第i天  持有  股票所得现金

        # 如果第i天持有股票，dp[i][1] 可有两种状态推导出来：
        # 1. 第i-1天就有股票，保持现状 dp[i-1][1] 
        # 2. 第i天买入股票，-prices[i]
        # 所以，dp[i][1] = max(dp[i-1][1], -prices[i])

        # 如果第i天不持有股票，dp[i][0] 可有两种状态推导出来：
        # 1. 第i-1天就不持有股票，保持现在，dp[i-1][0]
        # 2. 第i-1天持有股票，并且第i天卖了股票，dp[i-1][1] + prices[i]
        # 所以，dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])

        # basecase dp[0][1] = -prices[0]; dp[0][0] = 0 ; 返回dp[n-1][0] 返回最后一天没有股票时的值
        
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]
        dp[0][1] = -prices[0]
        dp[0][0] = 0
        for i in range(1, n):
            dp[i][1] = max(dp[i-1][1], -prices[i])
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        return dp[n-1][0]
    
    	# 思路3：精简思路2
        n = len(prices)
        dp_i_0 = 0
        dp_i_1 = float('-inf')
        
        for i in range(0, n):
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]) # 前一天不持有，或者前一天持有，然后卖出
            dp_i_1 = max(dp_i_1, -prices[i]) # 前一天持有，或者买入
        return dp_i_0
```



#### [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

```python
# 问题描述：尽可能多的去交易（买卖各一次，算一笔交易，并且必须先买后卖）
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 思路2 dp
        n = len(prices)
        dp_i_0 = 0
        dp_i_1 = float('-inf')

        for i in range(0, n): # 注意次序从0开始，从1 就错了
            temp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]) # 前一天不持有，或者前一天持有，然后卖出
            dp_i_1 = max(dp_i_1, temp - prices[i]) # 前一天持有，或者买入
        return dp_i_0
```



#### [123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

```python
# 问题描述：最多可完成2笔交易（买卖各一次，算一笔交易，并且必须先买后卖）
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxK = 2
        n = len(prices)
        dp = [[[0 for _ in range(2)] for _ in range(maxK+1)] for _ in range(n)] # 注意初始化 最小单元是二维
        for i in range(n):
            for k in range(maxK, 0, -1):
                if i == 0:  # 初始化base case
                    dp[0][k][0] = 0
                    dp[0][k][1] = -prices[0]
                else:
                    dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
                    dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        return dp[n-1][maxK][0]
```



#### [188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

```python
# 问题描述：最多可完成k笔交易（买卖各一次，算一笔交易，并且必须先买后卖）
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        # 第一种情况：最多k次交易，k小于 len(prices) // 2
        if k < len(prices) // 2: 
            maxK = k
            dp = [[[0 for _ in range(2)] for _ in range(maxK+1)] for _ in range(n)] # 注意初始化 最小单元是二维
            for i in range(n):
                for k in range(maxK, 0, -1):
                    if i == 0:  # 初始化base case
                        dp[0][k][0] = 0
                        dp[0][k][1] = -prices[0]
                    else:
                        dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
                        dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
            return dp[n-1][maxK][0]
        
        # 第二种情况：无限次交易
        else: # 无限次交易
            n = len(prices)
            dp_i_0 = 0
            dp_i_1 = float('-inf')

            for i in range(0, n): # 注意次序从0开始，从1 就错了
                temp = dp_i_0
                dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]) # 前一天不持有，或者前一天持有，然后卖出
                dp_i_1 = max(dp_i_1, temp - prices[i]) # 前一天持有，或者买入
            return dp_i_0
```



#### [309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) 

```python
# 问题：尽可能多的交易，但是不能连续交易两次
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 思路 dp
        n = len(prices)
        dp_i_0 = 0
        # dp_i_1 = - prices[0] # 错了
        # 解释：第i天持有股票，一定是i-2 天前转移过来的，而不是i-1 ；
        # 所以第1天持有股票，不是第0‘不持有’，然后第1天买入得到的，而是第-1天，也就是负无穷
        dp_i_1 = float('-inf') 
        dp_pre_0 = 0 
        for i in range(0, n): # 注意次序从0开始，从1 就错了
            temp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]) # 前一天不持有，或者前一天持有，然后卖出
            dp_i_1 = max(dp_i_1, dp_pre_0 - prices[i]) # 前一天持有，或者买入
            dp_pre_0 = temp

        return dp_i_0
```



#### [714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/) 

```python
# 问题：尽可能的多进行交易，但是每次交易有手续费（购买的时候）
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        dp_i_0 = 0
        dp_i_1 = float('-inf') 
        
        for i in range(0, n): # 注意次序从0开始，从1 就错了
            temp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]) # 前一天不持有，或者前一天持有，然后卖出
            dp_i_1 = max(dp_i_1, temp - prices[i] - fee) # 前一天持有，或者买入
        return dp_i_0
```



#### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/) 

```python
# 问题：非负整数数组 nums = [1,2,3,1]，不能连续取两个数，求最大值
# 时空：O(n)，O(n)，使用滚动数组，O(n)，O(1)
class Solution: 
    def rob(self, nums: List[int]) -> int:
        # dp[i] 表示偷到nums[i](包括nums[i]) 所获得的最大金额
        # 有两种选择，如果选择偷nums[i]，则dp[i] = dp[i-2] + nums[i]；
        # 如果只是路过，并不偷窃，则dp[i] = dp[i-1]
        # 所以 dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        # 由状态转移方程可以看出，i 与 i-1、i-2 有关，所以，需要求出dp[0], dp[1]
        # dp[0], 表示偷到nums[0], dp[0] = nums[0]；dp[i] = max(nums[0], nums[1])
        if not nums:
            return 0 
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        n = len(nums)
        dp = [0 for _ in range(n)]
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        return dp[n-1]

```



#### [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/) 

```python
# 问题：非负整数"环形"数组 nums = [1,2,3,1]，最后一个的相邻是第一个，不能连续取两个数，求最大值
# 时空：O(n)，O(n)，使用滚动数组，O(n)，O(1)
class Solution:
    def helper(self, nums: List[int]) -> int:
        if not nums:
            return 0 
        if len(nums) == 1:
            return nums[0]
        n = len(nums)
        dp = [0 for _ in range(n)]
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        return dp[n-1]

    def rob(self, nums: List[int]) -> int:
        # 分三种情况：1. 包含首nums[0], 不包含尾nums[n-1]；2. 不包含首nums[0]，包含尾部[n-1]，3. 首尾均不包括，
        # 其实情况3已经包含在情况1，2 内了，所以就两种情况
        # 思路：分别调用198. 打家劫舍的函数两次，选择最大的结果
        n = len(nums)
        if not nums:
            return 0
        if n == 1:
            return nums[0]
        res1 = self.helper(nums[1:])
        res2 = self.helper(nums[0:n-1])
        return max(res1, res2)
```



#### [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/) 

```python
# 问题：非负整数二叉树，不能取相邻的两个，求最大值
# 思路1：带记忆的递归
class Solution:
    def __init__(self):
        self.visited = {} # 存储遍历到某个结点，对应的最大收益
    def rob(self, root: TreeNode) -> int:
        # 思路1 ，带记忆的递归 
        # 时空 O(n)， O(n)
        
        if not root:
            return 0
        if not root.left and not root.right:
            return root.val
        if root in self.visited.keys():
            return self.visited[root]
        # 选择偷父节点
        val1 = root.val
        if root.left:
            val1 += self.rob(root.left.left) + self.rob(root.left.right) # 跳过左孩子
        if root.right:
            val1 += self.rob(root.right.left) + self.rob(root.right.right) # 跳过右孩子
        
        # 不偷父节点
        val2 = self.rob(root.left) + self.rob(root.right) # 考虑父节点的左右孩子

        # 记录访问当前结点所获得的最大收益
        self.visited[root] = max(val1, val2)
        
        # 返回最大收益

        return max(val1, val2)

# 思路2：动态规划


```



#### [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)

```python
# kmp 算法
```

#### [214. 最短回文串](https://leetcode-cn.com/problems/shortest-palindrome/)

```python
# kmp 算法
```



#### [343. 整数拆分](https://leetcode-cn.com/problems/integer-break/)

```python
# 一个整数拆成两份或两份以上，使得乘机最大
# 思路1 动态规划
# dp[i] 表示拆分数字i，可以获得的最大乘积是dp[i]
# 解析：设j 是 一个分裂点（因为i要拆成两份或两份以上），j 将i分成了两份：j、(i-j)
#      j部分不进行分割，(i-j)部分有两种选择，dp[i-j] 或 (i-j)
# 初始化：dp[1] = 1, dp[2] = 1
dp = [1 for _ in range(N+1)]
for i in range(3,N+1):
    for j in range(1,i-1):
        dp[i] = max(dp[i], dp[i-j]*j, (i-j)*j)  
return dp[n]
# 思路2 数学规则:拆成尽可能多的3
```



#### [509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

```python
class Solution:
    def fib(self, n: int) -> int:
        # 定义并初始化dp数组
        # dp[i] 表示第i个斐波那契数
        dp = [1 for _ in range(n+1)]
        dp[0] = 0
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
```



#### [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

```python
# 思路同斐波那契数列
class Solution:
    def climbStairs(self, n: int) -> int:
        # 注意台阶和楼顶的差别
        # 定义并初始化dp
        # dp[i] 表示到达第i个台阶，一共有dp[i]种方式
        dp = [1 for _ in range(n+1)]
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

```



#### [746. 使用最小花费爬楼梯](https://leetcode-cn.com/problems/min-cost-climbing-stairs/)

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # setp1 定义dp并初始化
        # dp[i] 表示到第i个阶梯所需要的最小花费
        # base case：dp[0]
        # dp[i] 的含义 为 第i层的最低花费
        # 初始从01开始所以dp[0]= dp[1] = 0
        # 阶梯是0，n-1的，阶梯顶是n 所以，最终返回 dp[n]
        n = len(cost)
        dp = [0 for _ in range(n+1)]
        for i in range(2, n+1):
            dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
        return dp[n]
```



#### [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

```python
# 问题：左上到右下，一共多少条路径（无障碍）
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # step1 定义dp数组
        # dp[i][j] 表示走到dp[i][j] 一共有多少种选择
        # base case：dp[0][j] = 1 , dp[i][0] = 1
        # return dp[m-1][n-1]
        dp = [[1 for _ in range(n)] for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]
```



#### [63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

```python
# 问题：左上到右下，一共多少条路径（有障碍）
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # setp1 定义dp 和初始化
        # dp[i][j] 表示从(0,0) 到(i,j) 共有dp[i][j]条路径
        m = len(obstacleGrid)  # 行
        n = len(obstacleGrid[0]) # 列
        dp = [[0 for _ in range(n)] for _ in range(m)]

        # 错误思路：第一行，当有一个障碍物，后面的都不通了
        for i in range(m):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = 1
            else:
                break

        for j in range(n):
            if obstacleGrid[0][j] == 0:
                dp[0][j] = 1
            else:
                break
                
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] =  dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
```



#### [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

```python
# 问题：给定数字n，求以1...n 为节点组成的二叉搜索树共有多少种
# 思路：设n=3，所以结果是：头节点为1的结果 + 头节点为2的结果 + 头节点为3的结果
#      头节点为1的结果 = 左子树有0个结点的棵数 * 右节点有2个结点的棵数 
#      头节点为2的结果 = 左子树有1个结点的棵数 * 右节点有1个结点的棵数 
#      头节点为1的结果 = 左子树有2个结点的棵数 * 右节点有0个结点的棵数 
# dp[i] 表示数字i 一共有多少种二叉搜索树
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0 for _ in range(n+1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n+1):
            for j in range(1, i+1):
                dp[i] += dp[j-1]*dp[i-j]
        return dp[n]
```



### 背包

> 问题：
>
> 1. 为什么01背包和完全背包在使用一维数组时，01使用逆序，完全使用顺序？
>
>    `因为假如01背包是按照顺序遍历，前面的物品有可能使用多次`

> ```python
> # 01背包框架
> # 一个可装载重量为W的背包和N个物品, 第i个物品的重量为wt[i]，价值为val[i], 每个物品最多取一个，最多能装的价值是多少?
> def knapsack01_1(W,N,wt,val):
>     # dp[i][w] 含义：只装前i个物品，背包容量为w时，的最大价值
>     dp = [[0 for _ in range(W+1)] for _ in range(N+1)]
>     for i in range(1, N+1): # 外层是容量
>         for w in range(1, W+1): 
>             if w - wt[i-1] < 0: # 当前背包装不下，只能选择不装入
>                 dp[i][w] = dp[i-1][w]
>             else:
>                 dp[i][w] = max(dp[i-1][w],
>                                dp[i-1][w-wt[i-1]] + val[i-1])
>     return dp[N][W]
> 
> def knapsack01_2(W,N,wt,val):
>     # dp[w] 含义：背包容量为w时，的最大价值
>     dp = [0 for _ in range(W+1)]
>     for i in range(1, N+1):
>         for w in range(W,0, -1):
>             if w - wt[i-1] < 0: # 当前背包装不下，只能选择不装入
>                 dp[w] = dp[w]
>             else:
>                 dp[w] = max(dp[w],dp[w-wt[i-1]] + val[i-1])
>     return dp[W] # 注意是dp[W] 而不是dp[N]
> 
> def knapsack01_3(W,N,wt,val):
>     # dp[w] 含义：背包容量为w时，的最大价值
>     dp = [0 for _ in range(W+1)]
>     for i in range(1, N+1):
>         for w in range(W,wt[i-1]-1, -1):
>                 dp[w] = max(dp[w],dp[w-wt[i-1]] + val[i-1])
>     return dp[W] # 注意是dp[W] 而不是dp[N]
> 
> 
> ```

>```python
># 完全背包框架
># 一个可装载重量为W的背包和N个物品, 第i个物品的重量为wt[i]，价值为val[i], 每个物品可以取无限个，最多能装的价值是多少?
>def knapsackComplete1(W,N,wt,val):
>    # dp[i][w] 含义：只装前i个物品，背包容量为w时，的最大价值
>    dp = [[0 for _ in range(W+1)] for _ in range(N+1)]
>    for i in range(1, N+1):
>        for w in range(1, W+1):
>            if w - wt[i-1] < 0: # 当前背包装不下，只能选择不装入
>                dp[i][w] = dp[i-1][w]
>            else:
>                dp[i][w] = max(dp[i-1][w],
>                               dp[i][w-wt[i-1]] + val[i-1])
>    return dp[N][W]
>
>def knapsackComplete2(W,N,wt,val):
>    # dp[w] 含义：背包容量为w时，的最大价值
>    dp = [0 for _ in range(W+1)]
>    for i in range(1, N+1):
>        for w in range(1, W+1):
>            if w - wt[i-1] < 0: # 当前背包装不下，只能选择不装入
>                dp[w] = dp[w]
>            else:
>                dp[w] = max(dp[w],dp[w-wt[i-1]] + val[i-1])
>    return dp[W] # 注意是dp[W] 而不是dp[N]
>
>def knapsackComplete3(W,N,wt,val):
>    # dp[w] 含义：背包容量为w时，的最大价值
>    dp = [0 for _ in range(W+1)]
>    for i in range(1, N+1):
>        for w in range(wt[i-1], W+1):
>                dp[w] = max(dp[w],dp[w-wt[i-1]] + val[i-1])
>    return dp[W] # 注意是dp[W] 而不是dp[N]
>
>```
>
>



#### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

```python
# 问题：coins = [1, 2, 5], amount = 11，计算可以凑成总金额所需的最少的硬币个数？ 硬币可以取无限个
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # step1 定义dp 并初始化
        # dp[i][j] 表示：用前i个物品，容量为j时，所需要的最少个数
        m = len(coins) # m 表示行，一共m+1 行，从0开始
        dp = [[0 for _ in range(amount + 1)] for _ in range(m+1)]
        
        # tips:dp[0][0] 是0 而不是float('inf')
        for j in range(1,amount+1):
            dp[0][j] = float('inf')

        # step2 
        for i in range(1, m+1):
            for j in range(1, amount+1):
                if j < coins[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-coins[i-1]] + 1)
        return dp[m][amount] if dp[m][amount] != float('inf') else -1
```



#### [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

```python
# 问题：coins = [1, 2, 5], amount = 11，计算可以凑成总金额组合的个数？ 硬币可以取无限个
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        # step1 定义dp，并初始化
        # dp[i][j] 表示：当使用前i个物品时，背包容量为j时，有dp[i][j]种方法可以装满背包
        # base case ：dp[0][...] = 0, dp[...][0] = 1
        m = len(coins) # m表示行
        dp = [[0 for _ in range(amount+1)] for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = 1
        
        # step2 
        for i in range(1,m+1):
            for j in range(1, amount+1):
                if j < coins[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-coins[i-1]]
                    # tips：
                    # 1. dp[i][j] 由两种选择之和组成；
                    # 2. 用的是dp[j][j-coins[i-1]] 而不是 dp[i-1][j-coins[i-1]]
        return dp[m][amount]

```



#### [416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

```python
# 问题：整数数组，是否能拆分长两个子集，使得两个子集元素和相等
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        
        # step1 初始化
        total = sum(nums)
        target = total // 2
        if 2*target != total:
            return False 

        # 问题转化01背包：从nums 选择物品，是否能凑成target
        # dp[i][j]表示前i个物品，每个数只能用一次，
        # 是否使得这些数的和恰好等于 j 
        # base case 
        m = len(nums)
        dp = [[False for _ in range(target+1)] for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = True
        
        # step2 
        for i in range(1, m+1):
            for j in range(1, target+1):
                if j < nums[i-1]:
                    dp[i][j] = dp[i-1][j] # 至少是这个答案，如果 dp[i−1][j] 为真，直接计算下一个状态
                # elif j == nums[i-1]:
                #     dp[i][j] = True
                # elif j > nums[i-1]:
                #     dp[i][j] = dp[i-1][j-nums[i-1]]
                else:
                    dp[i][j] = dp[i-1][j] | dp[i-1][j-nums[i-1]]
        return dp[m][target]
```



#### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

```python
# 问题：给定n，若干个完全平方数（1，4，9，...）相加和为n，求数量最少
# 思路：转换成完全背包问题，322. 零钱兑换
class Solution:
 def numSquares(self, n: int) -> int:
        # 完全背包问题，每个物品不限制个数 
        nums = []
        i = 1 
        while i**2 <= n:
            nums.append(i**2)
            i += 1
        # step1 定义dp 并初始化
        # dp[i][j] 表示：用前i个物品，容量为j时，所需要的最少个数
        m = len(nums)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        for j in range(1, n+1):
            dp[0][j] = float('inf')
        for i in range(1, m+1):
            for j in range(1, n+1):
                if j < nums[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-nums[i-1]] + 1)
        return dp[m][n]

```

#### [494. 目标和](https://leetcode-cn.com/problems/target-sum/)

```python
# 问题：nums: [1, 1, 1, 1, 1], S: 3，使用 + 或者 - 
# 思路1：转换成 01背包问题， base case 初始化有问题，待解决？
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        # 思路：因为只能用 +，- 两种符号，所有将数组分成两部分：A，B
        # 递推关系：A - B = S，A + B = sum(nums) --> A = (sum + S)//2 
        total = sum(nums)
        if total < S:
            return 0
        if total == S:
            return 1
        target = (total + S) // 2
        if target*2 != total + S:
            return 0
        
        # 转换成目标时target，从nums 里选，每一个物品选最多选1次，有多少种凑成方法
        # dp[i][j] 表示：当前i个物品，背包容量为j时，有多少种方法装满背包
        # base case：有问题
        #　
        m = len(nums)
        dp = [[0 for _ in range(target+1)] for _ in range(m+1)]
        # for i in range(m+1):
        #     dp[i][0] = 2
        # dp = [0 for _ in range(target+1)]
    
        
        for i in range(1, m+1):
            for j in range(1, target+1):
                if j < nums[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
        return dp[m][target]
```



#### [474. 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/)

```python
# 问题：返回最多m个0和n个1的最大子集的大小。字符串数组，每个元素由0、1组成
# 思路：转成01 背包问题 ，三层for循环
```

#### [剑指 Offer 14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

```python
# 问题同：343 整数拆分
# 一个整数拆成两份或两份以上，使得乘机最大
# 思路1 动态规划
# dp[i] 表示拆分数字i，可以获得的最大乘积是dp[i]
# 解析：设j 是 一个分裂点（因为i要拆成两份或两份以上），j 将i分成了两份：j、(i-j)
#      j部分不进行分割，(i-j)部分有两种选择，dp[i-j] 或 (i-j)
# 初始化：dp[1] = 1, dp[2] = 1
dp = [1 for _ in range(N+1)]
for i in range(3,N+1):
    for j in range(1,i-1):
        dp[i] = max(dp[i], dp[i-j]*j, (i-j)*j)  
return dp[n]
```



#### [剑指 Offer 14- II. 剪绳子 II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        # step1 定义dp并初始化
        # dp[i] 拆分长度为i的绳子，可以获取的最大乘积
        dp = [1 for _ in range(n+1)]
        for i in range(3,n+1):
            for j in range(1, i-1):
                dp[i] = max(dp[i], dp[i-j]*j, (i-j)*j)
        return dp[n] % 1000000007
```



#### [377. 组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/)

```python
# 思路2：完全背包
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        # dp[i] ：对于给定的由正整数组成且不存在重复数字的数组，和为 i 的组合的个数
        dp = [0 for _ in range(target+1)]
        dp[0] = 1
        for i in range(1, target+1): # 背包容量
            for j in range(len(nums)): # 物品个数
                if i >= nums[j]:
                    dp[i] += dp[i-nums[j]]
        return dp[target] 
```



### [贪心算法](https://mp.weixin.qq.com/s/weyitJcVHBgFtSc19cbPdw)

> 思想：通过局部最优，推出整体最优（证明：归纳法，反正法）
>
> 贪心算法一般分为如下四步：
>
> - 将问题分解为若干个子问题
> - 找出适合的贪心策略
> - 求解每一个子问题的最优解
> - 将局部最优解堆叠成全局最优解



#### [435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

```python
# 先排序
```



#### [452. 用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)

```python
# 问题：nums = [[10,16],[2,8],[1,6],[7,12]]， 返回引爆所有气球的最少的弓箭数
```



#### [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

```python
# 问题：非负整数数组，判断是否能跳到最后一个目标
# 贪心法
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        rightmost = 0 # 表示当前i 向右最远能跳到的位置
        n = len(nums)
        for i in range(n):
            if i <= rightmost: # i可以跳到的位置，必须是小于rightmost的
                rightmost = max(rightmost, nums[i] + i)
                if rightmost >= n - 1: # 如果能跳到的最远距离超过 数组长度，返回True
                    return True
        return False # 否则返回False
```



#### [45. 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)

```python
# 问题：非负整数数组，跳到最后一个目标的最小跳跃次数
# 贪心
class Solution:
    def jump(self, nums: List[int]) -> int:
        rightMost = 0 
        n = len(nums)
        end = 0      # 记录每一步跳跃可以到的区间的最后一个元素，用于记录何时jumps+=1
        jumps = 0    # 记录跳跃次数
        for i in range(n-1):
            rightMost = max(rightMost, nums[i] + i) # 不用判断，因为题目假设总能跳到最后
            if end == i:
                jumps += 1
                end = rightMost
        return jumps 

```



#### [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

#### [354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)

#### [455. 分发饼干](https://leetcode-cn.com/problems/assign-cookies/)

#### [376. 摆动序列](https://leetcode-cn.com/problems/wiggle-subsequence/)

#### [1005. K 次取反后最大化的数组和](https://leetcode-cn.com/problems/maximize-sum-of-array-after-k-negations/)

```python
# 思路
# 1. 按照从小到大排序， 首先把负数全部变成正数，然后在重新排序，选择最小的非负数执行剩余次数
```



#### [134. 加油站](https://leetcode-cn.com/problems/gas-station/)

```python
# 贪心
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # total记录可获得的总油量-总油耗， cur记录当前油耗情况， ans记录出发位置
        total, cur, ans = 0, 0, 0
        for i in range(len(gas)):
            total += gas[i] - cost[i]  # 到达位置i，邮箱里的油量
            cur += gas[i] - cost[i]
            if cur < 0:                     # 油不够开到i站
                cur = 0                     # cur置零，在新位置重新开始计算油耗情况
                ans = i + 1                 # 将起始位置改成i+1
        return ans if total >= 0 else -1    # 如果获得的汽油的量小于总油耗，则无法环
                                            # 行一周返回 -1；反之返回ans

```



#### [135. 分发糖果](https://leetcode-cn.com/problems/candy/)

```python
# 贪心
```



#### [860. 柠檬水找零](https://leetcode-cn.com/problems/lemonade-change/)

```python
# 思路：先收钱，再找钱
```



#### [406. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

```python
# 输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
# 输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
# 思路：按照身高排序，从最小得开始排，如果身高相同，先排第二维度小的，每次找到一个元素得最终位置；



```



#### [763. 划分字母区间](https://leetcode-cn.com/problems/partition-labels/)

```python
# 问题：由小写字母组成的字符串S，尽可能切成多分，任意两份没有重复字母
```



#### [738. 单调递增的数字](https://leetcode-cn.com/problems/monotone-increasing-digits/)

```python
# 问题：非负整数N，找到不大于N的“单调递增”数组，如N = 325，则输出299
# 思路：局部最优，当nums[i-1] > nums[i] 时，;
class Solution:
    def monotoneIncreasingDigits(self, N: int) -> int:
        # step1 数字转成数字数组
        nums = []
        while N:
            nums.append(N%10)
            N = N//10
        nums = nums[::-1]

        # step2 如果nums[i] > nums[i-1], 则nums[i-1] -= 1, nums[i] 变成9 ，用flag记录需要变成9的位置
        flag = len(nums)
        for i in range(len(nums)-1, 0, -1):
            if nums[i-1] > nums[i]:
                nums[i-1] -= 1
                flag = i

        # step3 根据flag 位置，将flag位置和以后的位置数字全变成9 
        for i in range(flag, len(nums)):
            nums[i] = 9
        
        # step4 将nums 转换成数字
        res = 0
        for one in nums:
            res = one + res*10 
        return res

```



### 滑动窗口

> **python知识点**：字典的get() 和setdefault() 区别：get()如果字典里没有，则返回一个默认值；setdefault()如果字典里没有，会创建健值对，默认值做为value
>
> **滑动窗口框架**
>
> ```python
> # 1. 初始化
> window = {}
> need = {} # 可以是字典，也可以是长度，根据题目条件初始化
> left = 0
> right = 0
> s = 'abcdefg'
> # 2. 开始滑动匹配
> while right < len(s):
>     
>     # 移动right索引
>     charRight = s[right]
>     window[charRight] = window.setdefault(charRight, 0) + 1
>     right += 1
> 	
>     # 判断窗口是否需要移动，移动left索引
>     while window need shrink:
>     	charLeft = s[charLeft]
>         window[charLeft] -= 1
>     	left += 1
> ```

#### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 1. 初始化
        windows = {}
        left = 0
        right = 0
        res = 0

        #２. 移动窗口
        while right < len(s):
            charRight = s[right]
            right += 1
            if charRight in windows.keys():
                windows[charRight] += 1
            else:
                windows[charRight] = 1
            while windows[charRight] > 1:
                charLeft = s[left]
                left += 1
                windows[charLeft] -= 1
            res = max(res, right - left)
        return res
```



#### [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # python 知识点：字典的get() 和setdefault() 区别：get()如果字典里没有，则返回一个默认值；
        # setdefault()如果字典里没有，会创建健值对，默认热最为key
    
        # 1. 初始化
        need = {}
        for char in t:
            need[char] = need.setdefault(char, 0) + 1 # 快速创建字典
        
        windows = {}
        left, right, valid = 0, 0, 0
        start = 0 
        length = len(s) + 1

        # 2. 滑动窗口
        while right < len(s):
            # 移动right索引
            charRight = s[right]
            right += 1
            if charRight in need:
                windows[charRight] = windows.setdefault(charRight, 0) + 1
                if windows[charRight] == need[charRight]:
                    valid += 1
            
            # 移动left索引
            while valid == len(need):
                if right - left < length:
                    start = left
                    length = right - left
                charLeft = s[left]
                left += 1
                if charLeft in need:
                    if windows[charLeft] == need[charLeft]:
                        valid -= 1
                    windows[charLeft] -=1 
            
        # 3. 返回最终结果
        return s[start:start+length] if length != len(s) + 1 else ''
```



#### [438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        
        #1. 初始化 need和windows 
        need = {}
        for char in p:
            need[char] = need.setdefault(char, 0) + 1
        window = {} # 滑动窗口
        left = 0
        right = 0
        valid = 0 # 记录匹配上的长度
        res = [] # 记录结果

        # 2. 开始滑动匹配
        while right < len(s):

            # 处理索引为right的字符
            charRight = s[right]
            right += 1
            if charRight in need:
                window[charRight] = window.setdefault(charRight, 0) + 1
                if window[charRight] == need[charRight]:
                    valid += 1

            # 判断窗口是否需要移动
            while (right - left) >= len(p):
                if valid == len(need):
                    res.append(left)
                charLeft = s[left]
                left += 1
                if charLeft in need:
                    if window[charLeft] == need[charLeft]:
                        valid -= 1
                    window[charLeft] -= 1
        return res
```



#### [567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/)

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # 1. 初始化 need 和window， s2 长 s， s1 短 t
        need = {}
        window = {}
        for char in s1:
            need[char] = need.setdefault(char, 0) + 1
        left, right, valid = 0, 0, 0

        # 2. 开始滑动匹配
        while right < len(s2):

            # 处理索引为right的字符，tips:right先+1
            charRight = s2[right]
            right += 1
            if charRight in need:
                window[charRight] = window.setdefault(charRight, 0) + 1
                if window[charRight] == need[charRight]:
                    valid += 1
            
            # 判断窗口是否需要移动，处理索引left
            while right - left >= len(s1):
                if valid == len(need):
                    return True
                charLeft = s2[left]
                left += 1
                if charLeft in need:
                    if window[charLeft] == need[charLeft]:
                        valid -= 1
                    window[charLeft] -= 1 # 这个位置总容易缩进。。。导致错误
        return False
```



#### [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

```python
# 问题：找出所有和为target的子数组，返回最短的那个
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # 1. 初始化条件
        if not nums:
            return 0
        numSum = 0 
        left = 0
        right = 0
        length = len(nums) + 1

        # 2. 开始滑动匹配
        while right < len(nums):
            
            # 处理索引为right的字符
            numSum += nums[right]
            right += 1
            
            # 判断窗口是否需要移动，处理索引为left的字符
            if numSum >= target and left <= right:
                while numSum >= target :
                    length = min(length, right - left)
                    numSum -= nums[left]
                    left += 1

        return length if length != len(nums) + 1 else 0

```

#### [395. 至少有K个重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/)

```python
# 思路：分治+递归
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        # 思路，每次递归，将字符串切切分成字串，
        # 递归截至条件：被分割的索引数组为空

        # 1. 建立字典
        hashMap = {}
        for char in s:
            hashMap[char] = hashMap.setdefault(char, 0) + 1

        # 2. 寻找切分点
        split = [] # 存放切分之后的子串起始索引
        for i in range(len(s)):
            if hashMap[s[i]] < k:
                split.append(i)

        # 3. 递归调用
        if len(split) == 0:
            return len(s) # 说明没有被切分，整个串满足条件
        split.append(len(s))
        res = 0
        left = 0
        for i in range(len(split)):
            l = split[i] - left
            if l > res:
                res = max(res, self.longestSubstring(s[left:left+l], k))
            left = split[i] + 1
        return res
```

#### [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

```python
# 滑动窗口，双指针
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        left = 1 # 或者 0
        right = 1 # 或者 0
        sum = 0 # 滑动窗口内的总和
        res = [] # 保存结果
        # while left <= target//2:
        while right <= target//2 + 2: # 循环范围和 left，right 初始化有关系
            if sum < target:
                sum += right
                right += 1
            elif sum > target:
                sum -= left
                left += 1
            else:
                arr = list(range(left,right)) # 范围到底是（left，right） 还是（left，right+1）也跟初始化有关
                res.append(arr)
                sum -= left
                left += 1
        return res 
            
```




### 双指针

> **python知识点**
>
> 1. 问题：某些情况下，for循环的结果会缺失一个；应该在循环之后再加一步处理
> 2. 字符串可以直接比较大小：print('a' >'b') ,  return False

#### [27. 移除元素](https://leetcode-cn.com/problems/remove-element/)

```python
# 问题：原地删除所有数值等于val的元素
# 使用快慢指针：slow一直指向等于val的元素，fast找到不等于val的元素，燃烧
def removeElement(nums, val):
    slow = 0
    for fast in range(len(nums)):
        # 如果nums[fast]、nums[slow] 都不等于 val，连个指针同时向后移动，直到slow指向元素等于val的节点。
        if nums[fast] != val:
            nums[slow] = nums[fast] # 先赋值
            slow += 1 # 后移动
   	return slow # slow 是下标索引，slow == 5， 说明slow前面有5个元素不等于val 

```



#### [26. 删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

```python
# 问题：排序数组，重复的只保留一个
# 思路使用快慢指针：slow、fast，
# 1.当nums[fast] == nums[slow]，fast向后移动，slow原地不动；
# 2.当nums[fast] != nums[slow]，把nums[fast] 赋给 nums[slow], 同时，fast、slow都向后移动一位
def removeDuplicates(nums):
    slow = 0
    for fast in range(len(nums)):
        if nums[slow] != nums[fast]:
            slow += 1 #先移动
            nums[slow] = nums[fast] # 后赋值
    return slow + 1 # slow 指向的是去重后最后一个元素的下标索引，整个数组长度应该是slow + 1



# 扩展：排序数组，重复的全部删除

```

#### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # # 思路1 暴力解法， 超时
        n = len(height)
        res = 0
        for i in range(1, n-1):
            lMax = 0
            rMax = 0
            for j in range(i, n):
                rMax = max(rMax, height[j])
            for j in range(i, -1, -1):
                lMax = max(lMax, height[j])
            res += min(lMax, rMax) - height[i]
        return res
        
        # 单调栈
        ans = 0
        stack = []
        for i in range(len(height)):
            while stack and height[stack[-1]] < height[i]:
                cur = stack.pop()
                if stack:
                    left = stack[-1]
                    right = i
                    curHeight = min(height[right], height[left]) - height[cur]
                    ans += (right - left - 1) * curHeight
            stack.append(i)
        return ans 

        # 双指针
        l = 0
        r = len(height) - 1
        res = 0
        while l < r:
            minHeight = min(height[l], height[r])
            if minHeight == height[l]:
                l += 1
                while height[l] < minHeight:
                    res += minHeight - height[l]
                    l += 1
            else:
                r -= 1
                while height[r] < minHeight:
                    res += minHeight - height[r]
                    r -= 1
        return res
```

#### [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

```python
# 双指针
class Solution:
    def maxArea(self, height: List[int]) -> int:
        i , j , res = 0, len(height) - 1, 0
        while i < j: # 每次移动矮的那边，每次移动，更新最大面积
            if height[i] < height[j]: 
                res = max(res, height[i]*(j-i))
                i += 1
            elif height[i] >= height[j]:
                res = max(res, height[j]*(j-i))
                j -= 1
        return res
```



#### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

```python
# 问题：无序数组，返回两数和为target的下标索引
def twoSum(nums, target):
    hashTable = {}
    for i, num in enumerate(nums):
        if target - num in hashTable.keys():
            return[i,hashTable[target - num]]
        else:
            hashTable[num] = i # 下标索引为value
     return []
        
```



```c++
// 暴力解法
class Solution{
public:
    vector<int> twoSum(vector<int>& nums, int target){
        int n = nums.size();
        for (int i = 0; i < n; ++i){
            for (int j = i + 1; j < n; ++j){
                if (nums[i] + nums[j] == target){
                    return {i, j};
                }
            }
        }
        return {};
    }
};

// 使用哈希表
class Solution{
public:
    vector<int> twoSum(vector<int>& nums, int target){
        unordered_map<int, int> hashtable;
        for (int i = 0; i < nums.size(); ++i){
            auto it = hashtable.find(target - nums[i]);
            if (it != hashtable.end()){
                return {it->second, i};
            }
            hashtable[nums[i]] = i;
        }
        return {};
    }
};
```



#### [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

```python
# 问题：无重复的nums，是否存在三元组的和为0？如果有求出
# 思路1：双指针
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        pre = float('-inf')
        for i in range(len(nums)-2):
            cur = nums[i]
            if cur == pre:
                continue
            pre = cur 
            left = i + 1
            right = len(nums) - 1
            while left < right:
                if nums[left] + nums[right] + cur == 0:
                    res.append([cur, nums[left], nums[right]])
                    j = left
                    while nums[j] == nums[left] and left < right:
                        left += 1
                elif nums[left] + nums[right] + cur > 0:
                    right -= 1
                elif nums[left] + nums[right] + cur < 0:
                    left += 1
        return res


# 思路2：回溯法
```



#### [18. 四数之和](https://leetcode-cn.com/problems/4sum/)

```python
# 问题：无重复的nums，是否存在四元组和为target？如果有求出来
# 思路1：双指针
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        # 处理长度小于4情况
        n = len(nums)
        if n < 4:
            return []
        nums.sort()
        res = []
        for i in range(n-3):
            if i > 0 and nums[i] == nums[i-1]: # 去除重复，确保nums[i] 改变了
                continue 
            for j in range(i+1, n-2):
                if j > i+1 and nums[j] == nums[j-1]: # 去除重复，确保nums[j] 改变了
                    continue
                # 开始双指针 
                left = j + 1
                right = n - 1
                while left < right:
                    if nums[i] + nums[j] + nums[left] + nums[right] < target:
                        left += 1
                    elif nums[i] + nums[j] + nums[left] + nums[right] > target:
                        right -= 1
                    
                    elif nums[i] + nums[j] + nums[left] + nums[right] == target:
                        res.append([nums[i], nums[j], nums[left], nums[right]])

                        # 下面是重点
                        while left+1 < right and nums[left] == nums[left+1]: # left 向右侧移动
                            left += 1
                        while left < right-1 and nums[right] == nums[right-1]:# right 向左侧移动
                            right -= 1 
                        left += 1
                        right -= 1
        return res






# 思路2：回溯方法（超时），求所有满足条件的组合 
```



#### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # 迭代
        pre = None
        cur = head 
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre

        # 递归
        if not head or not head.next:
            return head
        p = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return p
```



#### [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

![142环形链表2.png](https://pic.leetcode-cn.com/3be69ecc0e8948a5c0d74edfaed34d3eb92768ab781c1516bf00e618621eda66-142%E7%8E%AF%E5%BD%A2%E9%93%BE%E8%A1%A82.png)

```python
# 链表是否有环推导：
# 默认知识点：相遇一定是在环内，并且slow没有走完一圈，fast至少走完一圈（n>=1）
# tips：判断是否有环的时候，每次行走2步、或者3步都是可以的
# tips：查找环入口时，fast指针每次只能走2步？
# slow = x + y; fast = x + y + n*(y + z); 2*slow = fast 
# x = (n-1)*(y+z) + z :即：选两个指针，分别从head 和相遇点出发，每次走一步，两个指针的相遇点就是入口
```

```python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        slow, fast = head, head
        while True:
            if not fast or not fast.next:
                return False
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
```



#### [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

   ```python
   class Solution:
       def detectCycle(self, head: ListNode) -> ListNode:
   
           # 第一步判断是否有环
           slow, fast = head, head
           while True:
               if not fast or not fast.next:
                   return None
               fast = fast.next.next
               slow = slow.next
               if slow == fast:
                   break
   
           # 第二步找到环的入口
           fast = head
           while fast != slow:
               fast = fast.next
               slow = slow.next
           return fast
   ```

   

#### [344. 反转字符串](https://leetcode-cn.com/problems/reverse-string/)

   ```python
   def reverseString(self, s: List[str]) -> None:
       left = 0
       right = len(s) - 1
       while right > left:
           s[left], s[right] = s[right], s[left]
           left += 1
           right -= 1
   ```

   

#### [剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

```python
#思路1：直接替换
def replaceSpace(self, s: str) -> str:
    res = []
    for i in s:
        res.append(i if i != " " else "%20")
        return "".join(res)
# 思路2：双指针
```



#### [151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

```python
# 问题：反转单词顺序:"the sky is blue" -> "blue is sky the"
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip() # 去除首尾空格
        left, right = len(s) - 1, len(s) - 1
        res = []
        while left >= 0:
            while left >= 0 and s[left] != ' ': # left 从后向前，寻找空格
                left -= 1
            res.append(s[left+1:right+1]) # 左侧left+1，因为left是空格位置；右侧right+1，因为切片是左闭右开区间
            while s[left] == ' ':
                left -= 1 # 跳过所有空格
            right = left # right 指向下一个单词的尾字符
        return ' '.join(res)
```



#### [82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

```python
# 问题：重复元素全部删除（链表有序）
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = TreeNode(0) # 创建哑结点
        dummy.next = head
        pre = dummy; cur = head
        while cur: # 当前节点存在 
            while cur.next and cur.val == cur.next.val: # 下一个节点存在，且与当前节点值重复
                cur = cur.next # 当前节点后移
            if pre.next == cur: # 前一个节点的后节点为当前节点，意味着当前节点未移动，且后一个节点不重复
                pre = pre.next # 前一个节点后移
            else: # 前一个节点的后节点不为当前节点，意味着当前节点移动，且后一个节点重复
                pre.next = cur.next 
            cur = cur.next
        return dummy.next
```



#### [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return None
        slow = head
        fast = head
        while fast: # 循环到最后，fast == None
            if slow.val != fast.val:
                slow.next = fast
                slow = slow.next
                fast = fast.next
            else:
                fast = fast.next
        slow.next = fast
        return head
```



#### [283. 移动零](https://leetcode-cn.com/problems/move-zeroes/)

```python
# 问题：所有0移动到素组最前面
# 思路1 排序 

# 思路2：双指针
def moveZeros(nums):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[fast], nums[slow] = nums[slow], nums[fast]
          	left += 1
     
```

#### [202. 快乐数](https://leetcode-cn.com/problems/happy-number/)

```python
# 思路1：hash表
class Solution:
    def helper(self, a):
        res = 0
        while a > 0:
            res += (a % 10)**2 
            a = a // 10
        return res
    def isHappy(self, n: int) -> bool:
        temp = []
        while n != 1 and n not in temp: # 因为可能有循环
            temp.append(n)
            n = self.helper(n)
        return n == 1

# 思路2：快慢指针，判断链表是否有环
class Solution:
    def helper(self, a):
        res = 0
        while a > 0:
            res += (a % 10)**2 
            a = a // 10
        return res
    def isHappy(self, n: int) -> bool:
        fast = n
        slow = n 
        while fast != slow and fast != 1:
            slow = self.helper(n)
            fast = self.helper(self.helper(n))
        return fast == 1
```



#### [977. 有序数组的平方](https://leetcode-cn.com/problems/squares-of-a-sorted-array/)

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        right = len(nums) - 1
        while right >= 0:
            a = nums[0] ** 2
            b = nums[right] ** 2
            if b > a:
                nums[right] = b
                right -= 1
            else:
                nums[0] = nums[right]
                nums[right] = a
                right -= 1
        return nums
                
```



#### [75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/)

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left, right = 0, len(nums) - 1
        i = 0
        while i <= right:
            if nums[i] == 0:
                nums[i], nums[left] = nums[left], nums[i]
                i += 1
                left += 1
            elif nums[i] == 2:
                nums[i], nums[right] = nums[right], nums[i]
                right -= 1
            else:
                i += 1
```



#### [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        l, r = 0, len(nums) - 1
        while l <= r:
            if nums[l] % 2 == 1 and nums[r] % 2 == 0:
                l += 1
                r -= 1
            elif nums[l] % 2 == 0 and nums[r] % 2 == 0:
                r -= 1
            elif nums[l] % 2 == 1 and nums[r] % 2 == 1:
                l += 1
            elif nums[l] % 2 == 0 and nums[r] % 2 == 1:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1 
        return nums
```



#### [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        ta, tb = headA, headB
        while ta != tb:
            ta = ta.next if ta else headB
            tb = tb.next if tb else headA
        return tb 

```



#### [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

```python

```





### 回溯问题（排列）

> 回溯三要素：路径、选择列表，结束条件
>
> ```python
> def backtrack(self, track, nums, res):
>  if len(track) == len(nums):
>      res.append(track)
>  for one in nums:
>      if one in track:
>          continue
>     	track.append(one)
>      self.backtrack(track, nums, res)
>      track.pop()
> ```
>
> **tips**：
>
> 1. **startIndex** ：加入startIndex 是为了生成的track里面的顺序，和nums里原始顺序一致
> 2. **sort**：为了元素在加入track时，判断是否大于track[-1]
>
> 3. **dict**：原始元素里，有重复元素

#### [剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

```python
# 问题：字符串全排列（有重复）
# 化简：有重复+取一次
class Solution:
    def backTrack(self, track, res, nums, tempDict, l):
        if len(track) == l:
            res.append(''.join(track[:]))
            return 
        for i in range(len(nums)):
            if tempDict[nums[i]] >= 1:
                track.append(nums[i])
                tempDict[nums[i]] -= 1
                self.backTrack(track, res, nums, tempDict, l)
                tempDict[nums[i]] += 1
                track.pop()


    def permutation(self, s: str) -> List[str]:
        tempDict = {}
        nums = []
        track = []
        res = []
        l = len(s)
        for one in s:
            if one not in tempDict.keys():
                tempDict[one] = 1
                nums.append(one)
            else:
                tempDict[one] += 1
        self.backTrack(track, res, nums, tempDict, l)
        return res

```



#### [剑指 Offer 45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

```python
# 问题：字符串数组，讲字符串排成最小数字，输入：[10,2]， 输出："102"
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if int(str(nums[i]) + str(nums[j])) > int(str(nums[j]) + str(nums[i])):
                    nums[i], nums[j] = nums[j], nums[i]
        res = ''
        for one in nums:
            res += str(one)
        return res
```



#### [31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

```python
# 问题：全排列的下一个，如果没有下一个，返回最小的那个
# 思路1，全排列，找顺序

# 思路2：用规则
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead
        """
        i = len(nums) - 2
        j = len(nums) - 1

        # 第一步，找到‘较小数’，‘较大数’，并交换‘较小数’，‘较大数’
        while i >= 0 and nums[i] >= nums[i + 1]:  # 找到左侧‘较小数’
            i -= 1
        if i >= 0: # 找到右侧，从右侧到左，第一个比‘较小数’大的数字
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]

        # 第二步，从nums[i+1]开始，到末尾，必须是升序
        left, right = i+1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```



#### [60. 排列序列](https://leetcode-cn.com/problems/permutation-sequence/)

```python
# 问题：无重复元素数组，[1,2,3...n] 的全排列的第k个是什么
# 化简：无重复 + 取一次
# 思路1，用求出所有全排列的方法时间会超时
# 思路2：用获取下一个排列，获取k-1次，因为原始数组就是第一个排列
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:

        i = len(nums) - 2
        j = len(nums) - 1

        # 第一步，找到‘较小数’，‘较大数’，并交换‘较小数’，‘较大数’
        while i >= 0 and nums[i] >= nums[i + 1]:  # 找到左侧‘较小数’
            i -= 1
        if i >= 0: # 找到右侧，从右侧到左，第一个比‘较小数’大的数字
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]

        # 第二步，从nums[i+1]开始，到末尾，必须是升序
        left, right = i+1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    def getPermutation(self, n: int, k: int) -> str:
        nums = list(range(1,n+1))
        while k > 1:
            self.nextPermutation(nums)
            k -= 1
        nums = list(map(lambda x:str(x), nums))

        return ''.join(nums)
```



#### [46. 全排列](https://leetcode-cn.com/problems/permutations/)

```python
# 问题：无重复元素数组，所有全排列
# tips：没有使用startIndex和sort
# 化简：无重复 + 取一次
class Solution:
    def backTrack(self, track, nums, res):
        if len(track) == len(nums):
            res.append(track[:])
            return 
        for i in range(len(nums)):
            if nums[i] in track:
                continue
            track.append(nums[i])
            self.backTrack(track, nums, res)
            track.pop()
    def permute(self, nums: List[int]) -> List[List[int]]:
        track = []
        res = []
        self.backTrack(track, nums, res)
        return res
```



#### [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

```python
# 问题：有重复元素数组，所有全排列
# 化简：有重复 + 取一次
# 思路：有重复：则构建字典，然后遍历所有key（也就是list(set())）
class Solution:
    def backTrack(self, track, nums, res, tempDict, l):
        if len(track) == l:
            res.append(track[:])
            return 
        for i in range(len(nums)):
            if tempDict[nums[i]] >=1 :
                track.append(nums[i])
                tempDict[nums[i]] -= 1
                self.backTrack(track, nums, res, tempDict, l)
                tempDict[nums[i]] += 1
                track.pop()
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        tempDict = {}
        for one in nums:
            if one not in tempDict.keys():
                tempDict[one] = 1
            else:
                tempDict[one] += 1
        track = []
        res = []
        l = len(nums)
        nums = list(set(nums)) # 关键点
        self.backTrack(track, nums, res, tempDict, l)
        return res
```






### 回溯问题（数独）

#### [37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

```python
# 假设数独有唯一的解，用回溯去接
class Solution:
    # 判断board[row][col]填入 val 是否合法
    def isValid(self, row, col, val, board):
        # 判断行是否有重复的
        for i in range(9):
            if board[row][i] == val:
                return False
        # 判断列是否有重复的
        for j in range(9):
            if board[j][col] == val:
                return False
        # 判断小的9宫格里是否有重复的
        startRow = (row // 3) * 3
        startCol = (col // 3) * 3
        for i in range(startRow, startRow + 3):
            for j in range(startCol, startCol + 3):
                if board[i][j] == val:
                    return False
        # 最终返回true
        return True
    
    def backTrack(self, board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != '.': # 不为. 说明已经有元素了，直接跳过
                    continue
                # 重点1：在 board[i][j] 位置上填入 1～9 
                for k in range(1,10):
                    # 注意：k是字符串 不是数字
                    k = str(k)
                    if self.isValid(i, j, k, board):
                        board[i][j] = k # 相当于 append()
                        if self.backTrack(board):
                            return True
                        board[i][j] = '.' # 相当于 pop()
                return False # 9个数字都是过，都不行 返回False
        return True

    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        self.backTrack(board)
```



#### [36. 有效的数独](https://leetcode-cn.com/problems/valid-sudoku/)

```python
# 问题：已经填入一部分，判断当前数独能不能做为数独的题（即：行、列、块有没有重复的数字）
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # 初始化三个字典数组，每个字典数组 有九个字典
        rows = [{} for _ in range(9)]
        coloums = [{} for _ in range(9)]
        boxes = [{} for _ in range(9)] # 解释：0~8 共9块

        # 一次遍历
        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num != '.':
                    num = int(num)
                    # 重点：记不住就画图，分别写出某一块 i和j的取值范围；
                    # 例如：第5块，i 属于 3~5 ，j 属于 6~8 ， boxIndex = 1*3 + 2 = 5 
                    boxIndex = (i//3)*3 + j // 3 
                    
                    # 更新字典里的值
                    rows[i][num] = rows[i].setdefault(num, 0) + 1 
                    coloums[j][num] = coloums[j].setdefault(num, 0) + 1
                    boxes[boxIndex][num] = boxes[boxIndex].setdefault(num, 0) + 1

                    # 检查vlue 是否大于1 
                    if rows[i][num] > 1 or coloums[j][num] > 1 or boxes[boxIndex][num] > 1:
                        return False
        return True
```



#### [51. N 皇后](https://leetcode-cn.com/problems/n-queens/)

```python
# 问题：n×n 的棋盘上，n个皇后，所有解法
class Solution:
    def isValid(self, row, col, chess):
        # 判断行有没有Q
        for i in range(row):
            if chess[i][col] == 'Q':
                return False

        # 判断当前坐标，右上角有没有Q
        i = row - 1
        j = col + 1
        while i >= 0 and j < len(chess):
            if chess[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        # 判断当前坐标，左上角有没有Q
        i = row - 1
        j = col - 1
        while i >= 0 and j >= 0:
            if chess[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        return True 
    
    def backTrack(self, n, row, chess, res):
        if row == n:
            res.append(["".join(chess[i]) for i in range(n)] ) # 正确
            # res.append(chess) # 错误
            return 
        for col in range(n):
            if self.isValid(row, col, chess):
                chess[row][col] = 'Q'
                self.backTrack(n, row+1, chess, res)
                chess[row][col] = '.'

    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        chess = [['.' for _ in range(n)] for _ in range(n)]
        self.backTrack(n, 0, chess, res)
        return res
```



#### [52. N皇后 II](https://leetcode-cn.com/problems/n-queens-ii/)

```python
# 问题：n×n 的棋盘上，n个皇后，所有解法的数量
class Solution:
    def isValid(self, row, col, chess):
        # 判断行有没有Q
        for i in range(row):
            if chess[i][col] == 'Q':
                return False

        # 判断当前坐标，右上角有没有Q
        i = row - 1
        j = col + 1
        while i >= 0 and j < len(chess):
            if chess[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        # 判断当前坐标，左上角有没有Q
        i = row - 1
        j = col - 1
        while i >= 0 and j >= 0:
            if chess[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        return True 
    
    def backTrack(self, n, row, chess, res):
        if row == n:
            res.append(["".join(chess[i]) for i in range(n)] ) # 正确
            # res.append(chess) # 错误
            return 
        for col in range(n):
            if self.isValid(row, col, chess):
                chess[row][col] = 'Q'
                self.backTrack(n, row+1, chess, res)
                chess[row][col] = '.'

    def totalNQueens(self, n: int) -> int:
        res = []
        chess = [['.' for _ in range(n)] for _ in range(n)]
        self.backTrack(n, 0, chess, res)
        return len(res)
```





### 回溯问题（子集）

> 子集：没有顺序

#### [78. 子集](https://leetcode-cn.com/problems/subsets/)

```python
# 问题：无重复数组，所有子集
# 思路：
class Solution:
    def backTrack(self, track, nums, res, startIndex):
        if len(track) <= len(nums):
            res.append(track[:])
        for i in range(startIndex, len(nums)):
            if nums[i] not in track:
                track.append(nums[i])
                self.backTrack(track, nums, res, i+1)
                track.pop()
    def subsets(self, nums: List[int]) -> List[List[int]]:
        track = []
        res = []
        self.backTrack(track, nums, res, 0)
        return res
```



#### [90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)

```python
# 问题：有重复数组，所有子集
class Solution:
    def backTrack(self, track, nums, res, startIndex):

        if track not in res:
            res.append(track[:])
            #　return 会出错
        for i in range(startIndex, len(nums)):
            track.append(nums[i])
            self.backTrack(track, nums, res, i+1)
            track.pop()
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        track = []
        res = []
        nums.sort() #不sort也会出错
        self.backTrack(track, nums, res, 0)
        return res
```



#### [473. 火柴拼正方形](https://leetcode-cn.com/problems/matchsticks-to-square/)

#### [698. 划分为k个相等的子集](https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/)

```python
class Solution:
    # nums：要遍历的数字
    # k：将数组拆分成k个非空子集， 当前还剩几个
    # target：每个非空子集的和
    # curSum：当前子集的和
    # visited：表示nums[i] 是否被遍历过了
    def backTrack(self, nums, target, start, curSum, k, visited):
        if k == 0: # check 1
            return True
        if curSum == target: # check 2
            return self.backTrack(nums, target, 0, 0, k-1, visited)

        for i in range(start, len(nums)):
            # 如果已经被访问了，或者，之前总和+nums[i] 大于target，则不能选取
            if visited[i] or curSum + nums[i] > target: 
                continue
            # 否则选取，将visited 设置为true
            visited[i] = True # make choice
            if self.backTrack(nums, target, i+1, curSum + nums[i], k, visited):
                return True
            visited[i] = False
        return False
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        total = sum(nums)
        target = total // k
        if k * target != total:
            return False
            
        visited = [False for _ in range(len(nums))]
        return self.backTrack(nums, target, 0, 0, k, visited);
```





### 回溯问题（分割）

#### [131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

```python
class Solution:
    # 判断是否为回文串
    def isPalindrome(self,sList):
        if len(sList) == 1:
            return True
        left = 0
        right = len(sList) - 1
        while left <= right:
            if sList[left] == sList[right]:
                left += 1
                right -= 1
            else:
                return False
        return True
    
    # 回溯
    def backTrack(self, track, res, lastStr):
        if len(lastStr) == 0: # 如果字符串没有了，就代表分割完了
            res.append(track[:])
            return 
        for i in range(len(lastStr)):
            # 将输入的字符串lastStr 切成两份，fist、last；如果fist不是回文串，跳过；否则将first 加入到track
            fist = lastStr[:i+1]
            last = lastStr[i+1:]
            if not self.isPalindrome(fist):
                continue
            track.append(fist)
            self.backTrack(track, res, last)
            track.pop()
          
    def partition(self, s: str) -> List[List[str]]:

        track = []
        res = []
        self.backTrack(track, res, s)
        return res
```

#### [132. 分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)

```python
# 思路1：回溯，用131题目的思路会超时
# 思路2：dp
class Solution:
    def minCut(self, s: str) -> int:
        # dp[i] 表示范围 s[0:i], 不包括s[i]，最少要分割的次数为dp[i]
        # 设 j 是 0 ~ i 的一个切分点； 如果 j ~i 是回文串，则 dp[i] = min(dp[i], dp[j] + 1)
        # base case dp[0] = -1 ,dp[1]表示s[0:1] 为回文最少切割次数，显然为0，  最总返回dp[n]

        n = len(s)
        dp = [i for i in range(-1, n)] # 注意初始化
        for i in range(1, n+1):
            for j in range(i):
                if s[j:i] == s[j:i][::-1]:
                    dp[i] = min(dp[i], dp[j] + 1)
        return dp[n]
```



#### [93. 复原IP地址](https://leetcode-cn.com/problems/restore-ip-addresses/)

```python
class Solution:
    def isLegal(self, s):
        if s.isdigit() and 0 <= int(s) <= 255:
            if len(s) >= 2 and s[0] == '0':
                return False
            return True
        return False
    
    # track 必须满足长度为四，
    def backTrack(self, track, res, s):
        if len(track) == 4 and len(s) == 0:
            res.append('.'.join(track))
            return 
        for i in range(len(s)):
            first = s[:i+1]
            last = s[i+1:]
            if not self.isLegal(first):
                continue
            track.append(first)
            self.backTrack(track, res, last)
            track.pop()

    def restoreIpAddresses(self, s: str) -> List[str]:
        if len(s) > 12:
            return []
        res = []
        track = []
        self.backTrack(track, res, s)
        return res
```



#### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

```python
# 思路1：回溯会超时
# 思路2：dp
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # dp[i] 表示 前i个字符，是否能拆分成字典里的单词， 字符串的表示形式是s[0:i-1]
        # 最终的返回结果是：dp[n] 即 前n个字符能否拆分
        n = len(s)
        dp = [False for _ in range(n+1)] # 长度为n+1 的 dp数组
        dp[0] = True
        for i in range(1, n+1):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict: # 注意子字符串是s[j:i]
                    dp[i] = True
                    break
        return dp[n]

```



#### [140. 单词拆分 II](https://leetcode-cn.com/problems/word-break-ii/)

```python
# 思路1：回溯
class Solution:
    def backTrack(self, track, s, res, index, wordDict):
        if index == len(s):
            res.append(' '.join(track))
            return
        for i in range(index, len(s)):
            if s[index:i+1] not in wordDict:
                continue
            self.backTrack(track + [s[index:i+1]], s, res, i+1, wordDict)

    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        res = []
        track = []
        self.backTrack(track, s, res, 0, wordDict)
        return res
```





### 回溯问题（组合）

> 组合：没有顺序，[1,2,3] 和[3,2,1] 是同一个组合
>
> tips：每个元素只选一次的时候，需要startIndex
>
> tips2：重复有两种情况
>
> 第一种情况：结果里同时出现[1,2,3]和[3,2,1]... 通过sort 可以解决
>
> 第二种情况：结果里重复出现相同元素，如出现两次[1,2,3]
>
> 
>
> tips : 避免重复的思路，先排序，新假如进来的元素必须不小于前面的元素
>
> tips：not > and > or

#### [39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

```python
# 问题：无重复正整数数组（每个元素可以无限制取），找出所有和为traget的组合
# 化简：无重复 + 无限取：sort，每次添加只能比末尾的大
class Solution:
    def backTrack(self, track, nums, res, target):
        if target == 0:
            res.append(track[:])
        for i in range(len(nums)):
            # 2.添加剪枝 条件：新添加的元素必须必须大于或等于track最后一个元素
            if target - nums[i] >= 0 and (len(track) == 0 or nums[i] >= track[-1]):
                track.append(nums[i])
                self.backTrack(track, nums, res, target-nums[i])
                track.pop()
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        track = []
        candidates.sort() # 1. 先排序
        self.backTrack(track, candidates, res, target)
        return res
```

#### [40. 组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)

```python
# 问题：有重复正整数数组（每个元素可以使用一次），找出所有和为traget的组合
# 化简：有重复 + 取一次
class Solution:
    def backTrack(self, track, candidates, numsDict, res, target, startIndex):
        if target == 0:
            res.append(track[:])
        
        for i in range(startIndex, len(candidates)):
            if numsDict[candidates[i]] > 0 and target - candidates[i] >= 0 and (len(track) == 0 or candidates[i] >= track[-1]):
                track.append(candidates[i])
                numsDict[candidates[i]] -= 1
                self.backTrack(track, candidates, numsDict, res, target - candidates[i], i)
                numsDict[candidates[i]] += 1
                track.pop()

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        numsDict = {}
        for one in candidates:
            numsDict[one] = numsDict.setdefault(one, 0) + 1
        candidates = sorted(list(set(candidates)))
        res = []
        track = []
        startIndex = 0 
        self.backTrack(track, candidates, numsDict, res, target, startIndex)
        return res
```

#### [216. 组合总和 III](https://leetcode-cn.com/problems/combination-sum-iii/)

```python
# 问题：无重复数组nums = [1,2,3,4,5,6,7,8,9] ，选出k个数字（每个元素可以使用一次），且和为n，所有组合
# 化简：无重复 + 取一次：sort + startIndex
class Solution:
    def backTrack(self, track, res, nums, k, target, startIndex):
        if len(track) == k and target == 0:
            res.append(track[:])
        for i in range(startIndex, len(nums)):
            if target - nums[i] >= 0 and (len(track) == 0 or nums[i] > track[-1]):
                track.append(nums[i])
                self.backTrack(track, res, nums, k, target-nums[i], i+1)
                track.pop()

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        nums = [i for i in range(1,10)] # 1. 相当于有序数组
        track = []
        res = []
        self.backTrack(track, res, nums, k, n, 0)
        return res

```

#### [377. 组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/)

```python
# 问题：无重复数字nums（每个元素可以无限次），找出和为target的所有组合
# 化简：无重复 + 无限次：sort + startIndex
# 上述方法超时，可用背包问题解决
# 思路1：回溯法：超时
class Solution:
    def backTrack(self, track, nums, res, target):
        if target == 0:
            res.append(track[:])
            return 
        for i in range(len(nums)):
            if target - nums[i] >= 0:
                track.append(nums[i])
                self.backTrack(track, nums, res, target - nums[i])
                track.pop()
        
    def combinationSum4(self, nums: List[int], target: int) -> int:
        track = []
        res = []
        self.backTrack(track, nums, res, target)
        return len(res)

# 思路2：完全背包
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        # dp[i] ：对于给定的由正整数组成且不存在重复数字的数组，和为 i 的组合的个数
        dp = [0 for _ in range(target+1)]
        dp[0] = 1
        for i in range(1, target+1): # 背包容量
            for j in range(len(nums)): # 物品个数
                if i >= nums[j]:
                    dp[i] += dp[i-nums[j]]
        return dp[target] 

```

#### [77. 组合](https://leetcode-cn.com/problems/combinations/)

```python
# 问题：数组[1,2,3....n] ,找出所有k个数的组合
# 使用了startIndex，否则有重复元素
class Solution:
    def backTrack(self, track, nums, k, res, startIndex):
        if len(track) == k:
            res.append(track[:])
        for i in range(startIndex, len(nums)):
            if nums[i] in track:
                continue
            track.append(nums[i])
            self.backTrack(track, nums, k, res, i+1)
            track.pop()
    def combine(self, n: int, k: int) -> List[List[int]]:
        track = []
        res = []
        nums = [i for i in range(1,n+1)]
        startIndex = 0
        self.backTrack(track, nums, k, res, startIndex)
        return res
```

#### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

```python
# 问题：n对括号(), 输出所有可能的括号组合
# 停止条件：右括号的个数大于左括号的个数
class Solution:
    def backTrack(self, track, left, right, res):
        # left 表示左括号剩余的个数
        # right 表示右括号剩余的个数
        if left == 0 and right == 0:
            res.append(track)
            return 
        if left > 0:
            track += '('
            # 技巧：放到如果变量放到递归函数里变换，则函数外面不用改变，否则需要在外面进行变化
            self.backTrack(track, left-1, right, res) 
            track = track[:-1]
        if right > left: # right 剩余的比left 多
            # track += ')'
            self.backTrack(track + ')', left, right-1, res)
    def generateParenthesis(self, n: int) -> List[str]:
        track = ""
        res = []
        left = n
        right = n
        self.backTrack(track, left, right, res)
        return res
```

#### [17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

```python
class Solution:
    def backTrack(self, track, index, maps, digits, res):
        if len(track) == len(digits):
            res.append(track)
            return 
        # num = digits[index]
        charList = maps[digits[index]]
        for char in charList:
            self.backTrack(track+char, index+1, maps, digits, res)
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        maps = {'2':'abc', '3':'def', '4':'ghi', '5':'jkl', '6':'mno', '7':'pqrs', '8':'tuv', '9':'wxyz'}
        res = []
        track = ''
        self.backTrack(track, 0, maps, digits, res)
        return res
```





### 二分查找

> ```python
> # 标准二分查找
> def binarySearch(nums, target):
>     left = 0
>     right = len(nums) - 1
>     while left <= right:
>         mid = left + (right - left) // 2 # 在循环里面
>         if nums[mid] < target:
>             left = mid + 1
>         elif nums[mid] > target:
>             right = mid - 1
>         elif nums[mid] == target:
>             return mid
>      return -1 
> ```
>
> ```python
> # 寻找左侧边界的二分查找
> def leftBound(nums, target):
>     left = 0
>     right = len(nums) - 1
>     while left <= right:
>         mid = left + (right - left) // 2
>         if nums[mid] < target:
>             left = mid + 1
>         elif nums[mid] > target:
>             right = mid - 1
>         elif nums[mid] == target:
>             right = mid -1 # 注意差别
>     if left >= len(nums) or nums[left] != target:
>         return -1
>     return left
> ```
>
> ```python
> # 寻找右侧边界的二分查找
> def rightBound(nums, target):
>     left = 0
>     right = len(nums) - 1
>     while left <= right:
>         mid = left + (right - left) // 2
>         if nums[mid] < target:
>             left = mid + 1
>         elif nums[mid] > target:
>             right = mid - 1
>         elif nums[mid] == target:
>             left = mid + 1
>     if right < 0 or nums[right] != target:
>         return -1
>     return right
> ```
>
> 

#### [704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

```python
# 标准二分
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            elif nums[mid] == target:
                return mid
        return -1 
        
```



#### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

```python
# 思路：分别寻找左右边界，left 和 right ，返回[left, right]
# 二分查找
class Solution:
    def leftBound(self, nums, target):
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            elif nums[mid] == target:
                right = mid - 1
        if left >= len(nums) or nums[left] != target:
            return -1
        return left 

    def rightBound(self, nums, target):
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            elif nums[mid] == target:
                left = mid + 1
        if right < 0 or nums[right] != target:
            return -1
        return right
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left = self.leftBound(nums, target)
        right = self.rightBound(nums, target)
        return [left, right]
```



#### [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

```python
# 问题：排序数组旋转，其中元素各不相同
# 思路：二分查找
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[0] <= nums[mid]: # 说明左侧是有序的
                if nums[0] <= target < nums[mid]: # target 在左侧
                    right = mid - 1
                else: # target 在右侧
                    left = mid + 1 
            elif nums[mid] <= nums[-1]: # 说明右侧是有序的
                if nums[mid] < target <= nums[-1]: # target 在右侧
                    left = mid + 1
                else: # 在左侧
                    right = mid - 1
        return -1
```



#### [81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)

```python
# 问题：旋转排序数组，有重复的元素，判断target是否在数组里
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return True

            if nums[mid] > nums[left]: # 左半段
                if nums[left] <= target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            elif nums[mid] == nums[left]: # 不确定哪段，left ++
                left += 1

            elif nums[mid] < nums[left]: # 右半段，写成nums[mid] < nums[right] 会报错
                if nums[mid] <= target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return False
```



#### [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

```python
# 问题：旋转排序数组，没有重复元素，找最小
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]

        left = 0
        right = len(nums) - 1

        # 判断是否是单调的
        if nums[left] < nums[right]:
            return nums[left]

        # 处理旋转情况
        while left <= right:
            mid = left + (right - left) // 2

            # 如果mid ，mid-1， mid+1， 有一个刚好是旋转点
            if nums[mid] > nums[mid+1]:
                return nums[mid+1]
            elif nums[mid-1] > nums[mid]:
                return nums[mid]

            # mid 在一侧内
            elif nums[0] < nums[mid]: # 说明左侧有序，最小值在右侧
                left = mid + 1
            else:
                right = mid - 1
```



#### [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        left = 0
        right = x 
        res = 0
        while left <= right:
            mid = (left+right)//2
            if mid**2 <= x:
                res = mid 
                left = mid + 1
            else:
                right = mid -1
        return res
```

#### [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)    

```python
# 扩展：写出基础函数，即：两个有序数组，寻找第k小的数 ，当k == (m+n)//2, 即为中位数
class Solution:
    # 获取两个有序数组第k小的数字
    def getKthElement(self, nums1, nums2, k):
        m = len(nums1)
        n = len(nums2)
        index1, index2 = 0, 0
        while True:
            # 特殊情况
            if index1 == m:
                return nums2[index2 + k - 1]
            if index2 == n:
                return nums1[index1 + k - 1]
            if k == 1:
                return min(nums1[index1], nums2[index2])

            # 正常情况
            newIndex1 = min(index1 + k // 2 - 1, m - 1)
            newIndex2 = min(index2 + k // 2 - 1, n - 1)
            pivot1, pivot2 = nums1[newIndex1], nums2[newIndex2]
            if pivot1 <= pivot2:
                k -= newIndex1 - index1 + 1
                index1 = newIndex1 + 1
            else:
                k -= newIndex2 - index2 + 1
                index2 = newIndex2 + 1
	
    # 获取两个有序数组中位数
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m, n = len(nums1), len(nums2)
        totalLength = m + n
        # 总数为奇数时
        if totalLength % 2 == 1: 
            return self.getKthElement(nums1, nums2, (totalLength + 1) // 2)
        # 总数为偶数时
        else:
            return (self.getKthElement(nums1, nums2, totalLength // 2) + self.getKthElement(nums1, nums2, totalLength // 2 + 1)) / 2

```



### [排序](https://www.jianshu.com/p/bbbab7fa77a2)

> **排序知识点**：每轮结束后，有元素会在最终位置的排序算法：所以有简单选择排序、快速排序、冒泡排序、堆排序

![](https://upload-images.jianshu.io/upload_images/4905573-7cb6ea087fd0add9?imageMogr2/auto-orient/strip|imageView2/2/w/700/format/webp)

> ```python
> # 堆排序 三部曲，首先要调整、然后构建、然后进行堆排序
> # i为父结点，两个叶子结点分别为2*i+1， 2*i+2
> # i为叶子结点，父结点为（i-1）//2
> def heapify(arr,i,l):
>    	left = 2*i+1
>    	right = 2*i+2
>    	largest = i
>        if left < l and arr[left] > arr[largest]:
>        	largest = left
>    	if right < l and arr[right] > arr[largest]:
>        	largest = right
>     	if i != largest:
>        	arr[i], arr[largest] = arr[largest], arr[i]
>     	heapify(arr,largest,l)
> 
> # 只是构建大根堆，并不是排序，so：arr并未被排序
> def buildMaxHeap(arr,l):
> 	for i in range((l-1-1)//2,-1,-1):
> 		heapify(arr,i,l)
> 
> def maxHeapSort(arr,l):
> 	buildMaxHeap(arr, l)
> 	for j in range(l-1,-1,-1):
> 		arr[0], arr[j] = arr[j], arr[0]
> 		heapify(arr,0,j) #　重点
> 	return arr
> 
> arr = [3,2,5,5,0,7]
> arr = maxHeapSort(arr,len(arr))
> print(arr)
> 
> ```
>
> ```python
> # 归并排序
> def mergesort(seq):
> 	"""归并排序"""
> 	if len(seq) <= 1:
> 		return seq
> 	mid = len(seq) // 2  # 将列表分成更小的两个列表
> 	# 分别对左右两个列表进行处理，分别返回两个排序好的列表
> 	left = mergesort(seq[:mid])
> 	right = mergesort(seq[mid:])
> 	# 对排序好的两个列表合并，产生一个新的排序好的列表
> 	return merge(left, right)
> 
> def merge(left, right):
> 	"""合并两个已排序好的列表，产生一个新的已排序好的列表"""
> 	result = []  # 新的已排序好的列表
> 	i = 0  # 下标
> 	j = 0
> 	# 对两个列表中的元素 两两对比。
> 	# 将最小的元素，放到result中，并对当前列表下标加1
> 	while i < len(left) and j < len(right):
> 		if left[i] <= right[j]:
>    			result.append(left[i])
>    			i += 1
> 	else:
>    		result.append(right[j])
>    		j += 1
> 	result += left[i:]
> 	result += right[j:]
> 	return result
> 
> seq = [5,3,0,6,1,4]
> print '排序前：',seq
> result = mergesort(seq)
> print '排序后：',result
> ```
>
> ```python
> # 快速排序
> def quick_sort(lists,i,j):
>     if i >= j:
>         return list
>     pivot = lists[i]
>     low = i
>     high = j
>     while i < j:
>         while i < j and lists[j] >= pivot:
>             j -= 1
>         lists[i]=lists[j]
>         while i < j and lists[i] <=pivot:
>             i += 1
>         lists[j]=lists[i]
>     lists[j] = pivot
>     quick_sort(lists,low,i-1)
>     quick_sort(lists,i+1,high)
>     return lists
> nums = [5,0,7,3,2,5]
> print(quick_sort(nums, 0, len(nums)-1))
> 
> ```
>
> ```python
> # 插入排序
> 
> ```
>
> ```python
> # 计数排序
> # 缺点：当数值中有非整数时，计数数组的索引无法分配
> def countSort(nums):
>  # 找出最大最小值
>  minNum = min(nums)
>  maxNum = max(nums)
>  res = []
>  # 统计每个素出现的次数
>  countList = [0 for _ in range(maxNum-minNum+1)]
>  for one in nums:
>      countList[one-minNum] += 1
>  # 排序
>  for i, num in enumerate(countList):
>      while num != 0:
>          res.append(i + minNum)
>          num -= 1
>  return res
> 
> 
> ```
>
> ```python
> # 桶排序：但可以解决非整数的排序
> ```
>
> 



#### [剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

```python
# 思路：构建小顶堆
class Solution:
    def heapify(self, arr, i, l):
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < l and arr[left] < arr[smallest]:
            smallest = left
        if right < l and arr[right] < arr[smallest]:
            smallest = right
        if smallest != i:
            arr[smallest], arr[i] = arr[i], arr[smallest]
            self.heapify(arr, smallest, l)
    
    def buildMinHeap(self, arr, l):
        for i in range((l-1-1)//2, -1, -1):
            self.heapify(arr, i, l)
    
    def heapSort(self, arr, l, k):
        self.buildMinHeap(arr, l)
        for j in range(l-1, -1, -1):
            k -= 1
            if k < 0:
                break
            arr[0], arr[j] = arr[j], arr[0]
            self.heapify(arr, 0, j)

    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        self.heapSort(arr, len(arr), k)
        if k == 0:
            return []
        return arr[-k:]
# 快排(原来保持原来序列)
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if k >= len(arr):
            return arr
        def quickSort(arr, l, r):
            if l >= r:
                return arr
            i, j = l, r
            pivot = arr[i]
            while i < j:
                while i < j and arr[j] > pivot:
                    j -= 1
                arr[i] = arr[j]
                while i < j and arr[i] <= pivot:
                    i += 1
                arr[j] = arr[i]
            arr[i] = pivot
            if k < i:
                return quickSort(arr, l, i-1)
            if k > i:
                return quickSort(arr, i+1, r)
            return arr
        arr = quickSort(arr, 0, len(arr)-1)

        return arr[:k]
```





#### [295. 数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)

```python
# 堆排序
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.maxHeap = []
        self.minHeap = []

    def addNum(self, num: int) -> None:
        if len(self.maxHeap) == len(self.minHeap):
            heapq.heappush(self.minHeap, -heapq.heappushpop(self.maxHeap, -num))
        else:
            heapq.heappush(self.maxHeap, -heapq.heappushpop(self.minHeap, num))

    def findMedian(self) -> float:
        if len(self.maxHeap) == len(self.minHeap):
            return (-self.maxHeap[0] + self.minHeap[0]) / 2
        else:
            return self.minHeap[0]s
```



#### [480. 滑动窗口中位数](https://leetcode-cn.com/problems/sliding-window-median/)

```python
# 思路 二分查找
```



#### [912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

```python
# 默写快排， 原地修改
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def quickSort(arr, left, right):
            if left >= right:
                return arr
            i, j, pivot = left, right, arr[left]
            while i < j:
                while i < j and arr[j] > pivot:
                    j -= 1
                arr[i] = arr[j]
                while i < j and arr[i] <= pivot:
                    i += 1
                arr[j] = arr[i]
            arr[i] = pivot
            quickSort(arr, left, i-1)
            quickSort(arr, i+1, right)
            return arr 
        return quickSort(nums, 0, len(nums)-1)
```



#### [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

```python
# 思路归并排序
class Solution:
    def __init__(self):
        self.ans = 0
    def merge(self, left, right):
        i = 0
        j = 0
        res = []
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1
                self.ans += (len(left)-i) # 原因：左侧中，从left[i] 开始到left[-1] 对right[j] 都能构成逆序对 

        res += left[i:]
        res += right[j:]
        return res

    def mergeSort(self, arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = self.mergeSort(arr[:mid])
        right = self.mergeSort(arr[mid:])
        return self.merge(left, right)
        
    def reversePairs(self, nums: List[int]) -> int:
        self.mergeSort(nums)
        return self.ans
```

#### [315. 计算右侧小于当前元素的个数](https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/)

```python
# 求每个元素的逆序对数
```



#### [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

```python
# 约瑟夫环问题
# 从第i个元素出发，走m步后得下标是(i+m-1)，由于是环，所以要对len(arr)取模，
# 所以i = (i+m-1) % len(arr)
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        i = 0 
        arr = list(range(n)) # 创建数组方法
        while len(arr) > 1:
            i = (i + m - 1) % len(arr)
            arr.pop(i)
        return arr[0]
```

#### [1046. 最后一块石头的重量](https://leetcode-cn.com/problems/last-stone-weight/)

```python

```

#### [1049. 最后一块石头的重量 II](https://leetcode-cn.com/problems/last-stone-weight-ii/)



### 缺失数字问题

> ```python
> print(ord('1')) # 输出字符的ASCII编码
> print(chr(49)) # 将ASCII编码转成字符
> print(3&2) # 与：两个都为1，结果为1
> print(3^3) # 异或：相同为0，不同为1
> print(3|4) # 或：有一个为1，结果为1
> print(bin(5)) # 输出数字的二进制字符串：0b101
> print(9>> 1) # 9的二进制 1001 右移动以为，变成 0100（4）
> print(9<< 1) # 9的二进制 1001 左移动一位，变成 10010（18）
> ```
>
> **python知识点**：
>
> 1. 异或运算”是不进位的二进制加法
>
> 2. 如果 `a ^ b = c` ，那么 `a ^ c = b` 与 `b ^ c = a` 同时成立
>
> 3. 交换两个变量的值：可以使用异或运算，也可以使用加法运算
>
> 4. | 基于异或运算                                    | 基于加减法                                      |
>    | ----------------------------------------------- | ----------------------------------------------- |
>    | `a = a ^ b` <br />`b = a ^ b` <br />`a = a ^ b` | `a = a + b`<br /> `b = a - b`<br /> `a = a - b` |
>
> 

#### [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

```python
# 问题：0 ≤ nums[i] ≤ n-1, 长度为n-1的递增数组，[0,n-1]中缺少一个数字，找出这个数字
# 思路：二分查找
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == mid: # 说明nums[:mid+1] 是不缺数字的 
                left = mid + 1
            else: # 说明nums[mid:] 是不缺数字的 
                right = mid - 1
        return left
```



#### [剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

```python
# 问题：求1~n十进制，1出现的次数
# 例如：n = 12，则1~n，1，2，3，4，5，6，7，8，9，10，11，12 共出现 5次 1
# 思路：找规律
# 时空：O(logn)，O(1)
# 分三种情况
class Solution:
    def countDigitOne(self, n: int) -> int:
        # 1. 初始化
        digit, res = 1, 0
        high, cur, low = n // 10, n % 10, 0
        # cur 从各位开始移动
        while high != 0 or cur != 0:

            # 2. 三种情况
            if cur == 0: 
                res += high * digit
            elif cur == 1: 
                res += high * digit + low + 1
            else: 
                res += (high + 1) * digit
            
            # 3. 更新
            low += cur * digit
            cur = high % 10
            high //= 10
            digit *= 10
            
        return res
```



#### [面试题 17.19. 消失的两个数字](https://leetcode-cn.com/problems/missing-two-lcci/)

```python
# 问题：1 ≤ nums[i] ≤ n，其中缺少两个数字，找出他们
# 解析：时空O(n)，O(1)
# step1:1至n 缺两个，所以len(nums) + 2 == n
# step2:sumOfTwo = sum(1..n) - sum(nums)
# step3:threshold = sumOfTwo/2 ,因为缺失的两个数不相等，所以一个<= threshold, 一个> threshold
# step4:只对小于等于threshold的元素求和，得到第一个缺失的数字:
# sum(1..threshold) - sum(nums中, 小于等于threshold的元素)
# step5:第二个缺失的数字: sumOfTwo - 第一个缺失的数字

class Solution:
    def missingTwo(self, nums: List[int]) -> List[int]:
        total = sum(nums)
        n = len(nums) + 2
        sumOfTwo = (1+n)*n//2 - total
        threshold = sumOfTwo // 2

        sumOfLeft = 0
        for i in nums:
            if i <= threshold:
                sumOfLeft += i
        first = (1+threshold)*threshold // 2 - sumOfLeft
        second = sumOfTwo - first
        return [first, second]

```



#### [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

```python
# 问题：非空整数数组，只有一个元素出现一次，其余均出现两次，找出出现一次的元素
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for one in nums:
            res = res^one
        return res
```



#### [137. 只出现一次的数字 II](https://leetcode-cn.com/problems/single-number-ii/)

```python
# 问题：非空整数数组，只有一个元素出现一次，其余均出现三次，找出出现1次的元素
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return (3*sum(set(nums)) - sum(nums))//2
```



#### [260. 只出现一次的数字 III](https://leetcode-cn.com/problems/single-number-iii/)

```python
# 问题：非空整数数组，只有两个元素出现一次，其余均出现两次，找出出现1次的元素
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        # 第一步 找出所有元素的异或，也就是两个出现单次元素的异或
        ret = 0
        for i in nums:
            ret ^= i
        div = 1
        # 第二步 找出异或结果从右到左，第一个位1的位置
        while div & ret == 0:
            div <<= 1 # 只有1位为1的二进制数字（1，2，4.。。。）（二进制分别是0001，0010，0100.。。）
        a, b = 0, 0
        for n in nums:
            if n & div: # 说明那个位1，为1
                a ^= n
            else:
                b ^= n
        return [a, b]
```



#### [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

```python
# 问题：非空整数数组，只有两个元素出现一次，其余均出现两次
# 思路：异或
# step1：所有元素异或，得到两个只出现1次的数字的异或值
# step2：在最终异或的结果中，任意找出为1的位，对所有数字分成两组：该位为1的一组，为0的一组
# step3：对两组分别异或，得了两个缺失的数字
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        # 第一步 找出所有元素的异或，也就是两个出现单次元素的异或
        ret = 0
        for i in nums:
            ret ^= i
        div = 1
        # 第二步 找出异或结果从右到左，第一个位1的位置
        while div & ret == 0:
            div <<= 1 # 只有1位为1的二进制数字（1，2，4.。。。）（二进制分别是0001，0010，0100.。。）
        a, b = 0, 0
        for n in nums:
            if n & div: # 说明那个位1，为1
                a ^= n
            else:
                b ^= n
        return [a, b]

```



#### [268. 丢失的数字](https://leetcode-cn.com/problems/missing-number/)

```python
# 问题：0 ≤ nums[i] ≤ n，一共n个数字，找出[0,n]范围内没有出现的那个数字
# 简化：缺一个
# 思路用异或：len(nums)^nums[i]^i
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        missing = n
        for i, num in enumerate(nums):
            missing ^= i ^ nums[i]
        return missing
```



#### [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

```python
# 问题：1 ≤ nums[i] ≤ n，一共n+1个数字，假设nums里只有一个重复数字，找出这个重复数字(可能出现2次或者多次)
# 思路：快慢指针
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # 此题特定情况：数字从1开始（下标索引从0开始），并且只有一个重复数字，所以建立i->nums[i] 的连接，一定有环
        # 所以次问题转化成：找链表有环的入口
        fast, slow = 0, 0
        t = 0
        while True:
            #　方便理解，写成两行，上一行时索引，下一行是nums
            # 0，1，2，3，4
            # 1，3，4，2，2 
            # fast 每次从上到下走两步，如fast起始是0，走了 0 ->1 和 1—>3 两步，fast 变成3
            # slow 每次从上到下走一步，如slow起始是0，走了 0 -> 1 一步，slow 变成1 
            fast = nums[nums[fast]]
            slow = nums[slow]
            t += 1
            if slow == fast: # 说明快慢指针相遇，接下来要找环的入口，思路同链表找环的入口
                fast = 0
                while nums[slow] != nums[fast]:
                    fast = nums[fast]
                    slow = nums[slow]
                return nums[slow]

```



#### [448. 找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)

```python
# 问题： 1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组， 一些出现两次，一些出现一次，找出[1,n]中没有出现过的数字
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        res = []
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i] <= n and nums[i] != nums[nums[i]-1]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        
        for i in range(n):
            if nums[i] != i + 1:
                res.append(i+1) # 注意差别
        return res

```

#### [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)

```python
# 问题：一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i] <= n and nums[i] != nums[nums[i]-1]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1 # 注意差别
        return n + 1
```



#### [442. 数组中重复的数据](https://leetcode-cn.com/problems/find-all-duplicates-in-an-array/)

```python
#  问题： 1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组， 一些出现两次，一些出现一次，找出出现两次的元素
class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        res = []
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i] <= n and nums[i] != nums[nums[i]-1]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        
        for i in range(n):
            if nums[i] != i + 1:
                res.append(nums[i]) # 注意差别
        return res
```



### 计算器问题

> **python知识点**：
>
> ```python
> print(-5/3) # -1.6666666666666667
> print(-5//3) # -2,向下取整
> print(int(-5/3)) # -1 ，只保留整数部分
> 
> print(5/3) # 1.6666666666666667
> print(5//3) # 1 ,向下取整
> print(int(5/3)) # 1 ，只保留整数部分
> 
> # 判断是否为某种数据类型
> isinstance(i, list) # 判断i是否为list
> isinstance(i, int)
> isinstance(i, str)
> 
> # 判断字符串是否为数字
> s.isdigit()
> 
> # 负数与负数异或：结果为正数
> # 负数与整数异或：结果为负数
> ```

#### [371. 两整数之和](https://leetcode-cn.com/problems/sum-of-two-integers/)

```python
# 问题：不用+，- 求两数之和 
# 思路：位运算
#1. a + b 的问题拆分为 (a 和 b 的无进位结果) + (a 和 b 的进位结果)
#2. 无进位加法使用异或运算计算得出
#3. 进位结果使用与运算和移位运算计算得出
#4. 循环此过程，直到进位为

class Solution:
    def getSum(self, a, b):
        mask = 2**32 - 1   #限定32位 ， 32位，每位都是1 
        while b & mask:     #有进位就一直计算
            carry = a & b   #获得进位
            a = a ^ b       #无进位加法
            b = carry << 1  #进位左移
        return a & mask if b > mask else a  #防止溢出


```



#### [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

```python
#字符串中：数字，+-*/，空格，小括号
```



#### [227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

```python
# 问题：+ - * / 空格 ，没有括号
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        num = 0
        sign = '+'
        for i in range(len(s)):
            if s[i].isdigit():
                num = num*10 + int(s[i])
            if s[i] in '+-*/' or i == len(s)-1:
                if sign == '+':
                    stack.append(num)
                elif sign == '-':
                    stack.append(-num)
                elif sign == '*':
                    stack.append(stack.pop() * num)
                else:
                    stack.append(int(stack.pop() / num))
                num = 0
                sign = s[i]
        return sum(stack)
```



#### [772. 基本计算器 III](https://leetcode-cn.com/problems/basic-calculator-iii/)

```python

```



#### [415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

```python
# 大数加法
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        i = len(num1) - 1
        j = len(num2) - 1
        carry = 0
        res = ''

        while i >= 0 or j >= 0 or carry > 0:
            n1 = int(num1[i]) if i >= 0 else 0
            n2 = int(num2[j]) if j >= 0 else 0
            cur = (n1 + n2 + carry) % 10 # 注意别写成了 //
            carry = (n1 + n2 + carry) // 10
            res += str(cur)
            i -= 1
            j -= 1

        return res[::-1]
```



#### [43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/)

```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        m = len(num1)
        n = len(num2)
        res = [0 for _ in range(m+n)]
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                mul = int(num1[i]) * int(num2[j])
                p1 = i + j
                p2 = i + j + 1
                sum = mul + res[p2] 
                res[p2] = sum % 10 #　余数
                res[p1] += sum // 10 # 进位 注意是 += 

        # 从左到右，找到第一个不是0的位
        ans = 0
        j = 0
        for i in range(len(res)-1, -1, -1):
            ans += res[i] * 10**j         
            j += 1
        return str(ans)
```



#### [29. 两数相除](https://leetcode-cn.com/problems/divide-two-integers/)

```python
# 问题：不用乘法、除法、取mod运算
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:

        sign = 1 if dividend ^ divisor >= 0 else -1
        dividend = abs(dividend)
        divisor = abs(divisor)

        res = 0
        while dividend >= divisor:                  # 例：1023 / 1 = 512 + 256 + 128 + 64 + 32 + 16 + 8 + 4 + 1
            cur = divisor
            multiple = 1
            while cur + cur < dividend:             # 用加法求出保证divisor * multiple <= dividend的最大multiple
                cur += cur                          # 即cur分别乘以1, 2, 4, 8, 16...2^n，即二进制搜索
                multiple += multiple
            dividend -= cur
            res += multiple


        res = res if sign == 1 else -res
        return min(max(-2**31, res), 2**31-1)
```



#### [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

```python
# 问题：输入: [1,2,3,4]，输出: [24,12,8,6]，不能使用除法
# 时空：O(n), O(n)
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        l = [0 for _ in range(n)]
        r = [0 for _ in range(n)]
        ans = [0 for _ in range(n)]
        
        # 求每个元素左侧的乘积
        l[0] = 1
        for i in range(1,n):
            l[i] = nums[i-1] * l[i-1]

        # 求每个元素右侧的乘积
        r[n-1] = 1
        for i in range(n-2, -1, -1):
            r[i] = nums[i+1] * r[i+1]
        
        # 求左右乘积
        for i in range(n):
            ans[i] = l[i] * r[i]
        return ans
```

#### [311. 稀疏矩阵的乘法](https://leetcode-cn.com/problems/sparse-matrix-multiplication/)

```python
class Solution:
    def multiply(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        res = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

        for i in range(len(A)):
            for j in range(len(B[0])):
                cur = 0
                for k in range(len(B)):
                    cur += A[i][k]*B[k][j] # 重点
                res[i][j] = cur
        return res
```





### [岛屿问题](https://leetcode-cn.com/problems/number-of-islands/solution/dao-yu-lei-wen-ti-de-tong-yong-jie-fa-dfs-bian-li-/)

> 岛屿问题的深度遍历（dfs）两要素：访问相邻结点，判断base case
>
> ```python
> def dfs(grid, r, c):
>  # 判断base case，如果超出了范围， 直接返回
>  if inArea(grid, r, c):
>    	return 
> 
> 
>  # 如何避免重复遍历：标记已经遍历过的格子
>  # 0 —— 海洋格子
>  # 1 —— 陆地格子（未遍历过）
>  # 2 —— 陆地格子（已遍历过）
> 
>  # 如果这个格子不是岛屿（0是海，2是遍历过了），直接返回
>  if grid[r][c] != 1:
>    	return 
> 
>  # 遍历格子为1的岛屿，并将格子改为：已遍历过
>  grid[r][c] == 2
> 
>  #　访问上下左右四个相邻结点
>  dfs(grid, r+1, c)
>  dfs(grid, r-1, c)
>  dfs(grid, r, c-1)
>  dfs(grid, r, c+1)
> 
> # 判断(r, c)是否再grid里面
> def inArea(grid, r, c):
>  return r <= 0 and r < len(grid) and c <= 0 and c < len(grid[0])
> ```
>
> 

#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

```python
class Solution:
    def inArea(self, grid, r, c):
        return 0 <= r < len(grid) and 0 <= c < len(grid[0])
    
    def findArea(self, grid, r, c):
        if not self.inArea(grid, r, c) or grid[r][c] == '0':
            return 
        grid[r][c] = '0'
        self.findArea(grid, r+1, c)
        self.findArea(grid, r-1, c)
        self.findArea(grid, r, c-1)
        self.findArea(grid, r, c+1)

    def numIslands(self, grid: List[List[str]]) -> int:
        res = 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == '1':
                    res += 1
                    self.findArea(grid, r, c)
        return res
```



#### [463. 岛屿的周长](https://leetcode-cn.com/problems/island-perimeter/)

```python
# 思路：一片水域上只有一个岛屿
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        res = 0 # 最终的周长
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1: # 遇到岛屿
                    cur = 0 # 表示对周长的贡献
                    # 上
                    if i -1 >= 0 and grid[i-1][j] == 0 or i -1 < 0:
                        cur += 1

                    # 下
                    if i + 1 <= len(grid) - 1 and grid[i+1][j] == 0 or i + 1 > len(grid) - 1 :
                        cur += 1

                    # 左
                    if j - 1 >= 0 and grid[i][j-1] == 0 or j - 1 < 0:
                        cur += 1

                    # 右
                    if j + 1 <= len(grid[0]) - 1 and grid[i][j+1] == 0 or j + 1 > len(grid[0]) - 1:
                        cur += 1
                        
                    res += cur 
        return res
```



#### [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

```python
class Solution:
    #  首先有个岛屿，是二维数组，grid，然后判断给定的点是否在这个岛屿中
    def inArea(self, grid, r, c):
        return 0 <= r < len(grid) and 0 <= c < len(grid[0])

    def findMaxArea(self, grid, r, c):
        if not self.inArea(grid, r, c) or grid[r][c] == 0:
            return 0
        grid[r][c] = 0 # 原来是1，遍历之后把它置为0   
        return 1 + self.findMaxArea(grid, r+1, c) + self.findMaxArea(grid, r-1, c) + self.findMaxArea(grid, r, c+1) + self.findMaxArea(grid, r, c-1)

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        maxArea = 0 
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:     
                    maxArea = max(maxArea, self.findMaxArea(grid, r, c))
        return maxArea
```



#### [827. 最大人工岛](https://leetcode-cn.com/problems/making-a-large-island/)

```python
# 思路：1. 先遍历每个岛屿，将每个岛屿上的元素赋值为岛屿的面积，并且给每个岛屿进行编号；2. 遍历grid[r][c] == 0的结点， 将两个不同编号的
```



#### [剑指 Offer 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

```python
class Solution:
    def helper(self, x):
        res = 0
        while x > 0:
            print(x)
            res += x % 10
            x = x // 10
        return res

    def islegal(self, r, c, m, n, k):
        return 0 <= r < m and 0 <= c < n and (self.helper(r) + self.helper(c)) <= k

    def dfsSearch(self, r, c, m, n, k, visited):
        if not self.islegal(r, c, m, n, k) or visited[r][c] == True:
            return 0
        visited[r][c] = True
        return 1 + self.dfsSearch(r+1, c, m, n, k, visited)\
                + self.dfsSearch(r-1, c, m, n, k, visited)\
                + self.dfsSearch(r, c+1, m, n, k, visited)\
                + self.dfsSearch(r, c-1, m, n, k, visited)

    def movingCount(self, m: int, n: int, k: int) -> int:
        visited = [[False for _ in range(n)] for _ in range(m)]
        res = 0
        res = self.dfsSearch(0, 0, m, n, k, visited)
        return res
```



#### [79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

```python
class Solution:
    def find(self, board, word, index, i, j):
        # 如果当前遍历过的字符的个数大于word的长度，说明都遍历完了，符合条件
        if index >= len(word):
            return True
         # 判断输入i，j是否合法
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
            return False
        if board[i][j] != word[index]:
            return False
        # 当前遍历的字符在单词里，修改board这个字符，防止遍历重复
        board[i][j] = '0'

        if self.find(board, word, index+1, i+1, j) or \
            self.find(board, word, index+1, i-1, j) or\
            self.find(board, word, index+1, i, j+1) or\
            self.find(board, word, index+1, i, j-1):
            return True
        board[i][j] = word[index]
        return False

    def exist(self, board: List[List[str]], word: str) -> bool:
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.find(board, word, 0, i, j):
                    return True
        return False
```

#### [73. 矩阵置零](https://leetcode-cn.com/problems/set-matrix-zeroes/)

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # 思路，先找到0元素的横纵坐标
        rows = []
        cols = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    rows.append(i)
                    cols.append(j)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i in rows or j in cols:
                    matrix[i][j] = 0
```



#### [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/)

```python
class Solution:
    # 感染函数，如果当前字符为O，把其上下左右的字符都变成O
    def infect(self, board, i, j):
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
            return 
        if board[i][j] != 'O':
            return
        board[i][j] = '-'
        self.infect(board, i+1, j)
        self.infect(board, i-1, j)
        self.infect(board, i, j+1)
        self.infect(board, i, j-1)

    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row = len(board)
        col = len(board[0])

        # step1 找到所有边界上的O，并把与其关联的O全部置为‘-’
        for i in range(row):
            self.infect(board, i, 0) # 第i行的第0个
            self.infect(board, i, col-1) # 第i行的最后一个
        
        for j in range(col):
            self.infect(board, 0, j) # 第0行的第j个
            self.infect(board, row-1, j) # 最后一行的第j个
        
        # step2 剩余所有的O都是被X所包围的，所以全部改成X; 
        # step3 同是所有‘-’都是原来没有被包围的O；将其还原
        for i in range(row):
            for j in range(col):
                if board[i][j] == 'O': # 顺序不能变
                    board[i][j] = 'X'
                if board[i][j] == '-':
                    board[i][j] = 'O'
```



#### [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/)

```python
class Solution:
    def dfs(self, i, j, record, matrix):
        rows, cols = len(matrix), len(matrix[0])
        if record[i][j] != -1:
            return record[i][j]
        length = 1 
        for (x, y) in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            x, y = x + i, y + j
            if 0<= x < rows and 0 <= y < cols and matrix[x][y] > matrix[i][j]:
                length = max(length, self.dfs(x, y, record, matrix) + 1)
        record[i][j] = length
        return length


    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        record = [[-1 for _ in range(len(matrix[0]))] for _ in range(len(matrix))] # 用来记录
        rows = len(matrix)
        cols = len(matrix[0])
        res = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                res = max(self.dfs(i, j, record, matrix), res)
        return res
```








### 二叉树（二叉搜索树）

>   ```python
>   # 1. 查找
>   
>   # 2. 插入
>   
>   # 3. 删除
>   ##1. 如果待删除的结点是叶子结点，直接删除
>   
>   ##2. 如果待删除的结点只有一个孩子，删除该结点，孩子替代当前结点
>   
>   ##3. 如果待删除的结点有两个孩子结点，删除该结点，用右子树最小的结点代替，递归删除右子树最小的结点（重复1，2，3）
>   ```
>   
>   

#### [230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

```python
# 思路1:中序遍历，取第k-1个
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        res = []
        def LNR(root, res):
            if not root:
                return 
            LNR(root.left, res)
            res.append(root.val)
            LNR(root.right, res)
            return res
        res = LNR(root, res)
        return res[k-1]
# 思路2 ：
```



#### [538. 把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)

```python
# 反序中序遍历
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        res = []
        stack = []
        total = 0
        pre = root
        while stack or root:
            if root:
                stack.append(root)
                root = root.right
            else:
                node = stack.pop()
                total += node.val
                node.val = total
                root = node.left
        return pre
```



#### [450. 删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst/)

#### [701. 二叉搜索树中的插入操作](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

```python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        # 递归
        if not root:
            return TreeNode(val)
        elif root.val > val:
            root.left = self.insertIntoBST(root.left, val)
        elif root.val < val:
            root.right = self.insertIntoBST(root.right, val)
        return root 

        # 迭代
        if not root:
            return TreeNode(val)
        pos = root
        while pos:
            if val < pos.val:
                if not pos.left:
                    pos.left = TreeNode(val)
                    break
                else:
                    pos = pos.left
            else:
                if not pos.right:
                    pos.right = TreeNode(val)
                    break
                else:
                    pos = pos.right
        return root
```



#### [700. 二叉搜索树中的搜索](https://leetcode-cn.com/problems/search-in-a-binary-search-tree/)

```python
# 思路：中序遍历
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        stack = []
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                node = stack.pop()
                if node.val == val:
                    return node
                root = node.right
        return None
```



#### [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        stack = []
        inorder = float('-inf')
        
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                root = stack.pop()
                # 如果中序遍历得到的节点的值小于等于前一个 inorder，说明不是二叉搜索树
                if root.val <= inorder:
                    return False
                inorder = root.val
                root = root.right
        return True
```



#### [530. 二叉搜索树的最小绝对差](https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/)

```python
class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        # 因为是非负，所以任意差值的最小值一定是相邻的两个结点
        res = []
        stack = []
        minres = float('inf')
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                node = stack.pop()
                if res:
                    minres = min(minres, abs(res[-1]-node.val))
                res.append(node.val)
                root = node.right
        return minres
```



#### [501. 二叉搜索树中的众数](https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/)

#### [235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

```python
# 思路 对于二叉搜索树，已知两个节点，寻找最近祖先节点，根据下面的递推关系式：
#1. 若 node.val < min(p.val,q.val)，则p和q的最近祖先节点一定在右子树；
#2. 若 max(p.val,q.val) < node.val，则p和q的最近祖先节点一定在左子树；
#3. 其他情况 node.val位于p.val和q.val间(可能等于node.val)，则node就是p和q的最近祖先。
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        # 递归
        if root.val < min(p.val, q.val):
            return self.lowestCommonAncestor(root.right, p, q)
        elif root.val > max(p.val, q.val):
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root
        
        # 迭代
        while root:
            if p.val > root.val and q.val > root.val:
                root = root.right
            elif p.val < root.val and q.val < root.val:
                root = root.left
            else:
                return root


```



#### [669. 修剪二叉搜索树](https://leetcode-cn.com/problems/trim-a-binary-search-tree/)

#### [108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)

```python
  class Solution:
      def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
          def helper(left, right):
              if left > right:
                  return None
  
              # 总是选择中间位置左边的数字作为根节点
              mid = (left + right) // 2
  
              root = TreeNode(nums[mid])
              root.left = helper(left, mid - 1)
              root.right = helper(mid + 1, right)
              return root
  
          return helper(0, len(nums) - 1)
```

  

#### [109. 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/)

```python
  class Solution:
      def sortedListToBST(self, head: ListNode) -> TreeNode:
          if not head:
              return None
          elif not head.next:
              return TreeNode(head.val)
          pre = None
          fast = head
          slow = head
          while fast and fast.next:
              pre = slow
              fast = fast.next.next
              slow = slow.next
          
          root = TreeNode(slow.val)
          pre.next = None # slow 前面的那个结点
  
          root.left = self.sortedListToBST(head)
          root.right = self.sortedListToBST(slow.next)
          return root
```

  

#### [1008. 前序遍历构造二叉搜索树](https://leetcode-cn.com/problems/construct-binary-search-tree-from-preorder-traversal/)

```python
 # 思路1：先排序，求出中序，然后根据 中序+前序 构造二叉树 
 
 
 
 # 思路2：前序遍历结果一定是：根 + 左侧遍历结果 + 右侧遍历结果，分别递归左右两侧
 class Solution:
     def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
         if preorder:
             root = TreeNode(preorder.pop(0))
             l, r = [], []
             for i in preorder:
                 if i <= root.val:
                     l.append(i)
                 else:
                     r.append(i)
             root.left = self.bstFromPreorder(l)
             root.right = self.bstFromPreorder(r)
             return root 
         else:
             return None 
```

 

#### [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

```python
   # 与114. 二叉树展开为链表，同看
   # 题目同 426 
 class Solution:
     def treeToDoublyList(self, root: 'Node') -> 'Node':
         if not root:
             return root
         
         head = Node(-1, None, None)# 当一个中间节点
         prev = head # 记录为先前节点,找到下一个节点才能串起来
 
         stack = []
         p = root
         while stack or root:
             if root:
                 stack.append(root)
                 root = root.left
             else:
                 root = stack.pop()
 
                 prev.right = root # 添加的1
                 root.left = prev # 添加的2
                 prev = root # 添加的3
 
                 root = root.right 
 
         head.right.left = prev
         prev.right = head.right
         return head.right
```

   

#### [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

```python
 class Solution:
     def flatten(self, root: TreeNode) -> None:
         """
         Do not return anything, modify root in-place instead.
         """
         if not root:
             return root
         self.flatten(root.left)
         self.flatten(root.right)
 
         leftChild = root.left
         rightChild = root.right
 
         root.left = None
         root.right = leftChild
 
         p = root
         while(p.right != None):
             p = p.right
         p.right = rightChild
```

 

#### [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

```python
 class Solution:
     def numTrees(self, n: int) -> int:
         dp = [0 for _ in range(n+1)]
         dp[0] = 1
         dp[1] = 1
         for i in range(2, n+1):
             for j in range(1, i+1):
                 dp[i] += dp[j-1]*dp[i-j]
         return dp[n]
```

 

#### [99. 恢复二叉搜索树](https://leetcode-cn.com/problems/recover-binary-search-tree/)

```python
 class Solution:
     def recoverTree(self, root: TreeNode) -> None:
         """
         Do not return anything, modify root in-place instead.
         """
         firstNode = None
         secondNode = None
         pre = TreeNode(float("-inf"))
 
         stack = []
         p = root
         while root or stack:
             if root:
                 stack.append(root)
                 root = root.left
             else:
                 root = stack.pop()
                 
                 if not firstNode and pre.val > root.val:
                         firstNode = pre
                 if firstNode and pre.val > root.val:
                     secondNode = root
                 pre = root
                 root = root.right
         firstNode.val, secondNode.val = secondNode.val, firstNode.val
```

 



### 二叉树（AVL）

#### [1382. 将二叉搜索树变平衡](https://leetcode-cn.com/problems/balance-a-binary-search-tree/)

#### [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)



### 二叉树（路径问题）

>**python知识点**：
>
>```python
>a = 1 # 全局变量 ，定义在整个文件的最外面
>def fun():
>    # 如果不声明 global a；全局变量是不能修改的，但是能访问
>    print(a)
>
>    # 如果想修改或访问全局变量，需要 使用global 定义，
>    global a
>    a += 1
>    print(a)
>    
>    
>```
>
>```python
># 函数嵌套使用：内层函数，使用外层函数的变量，使用关键字 nonlocal
>def outer():
>    x = "local"
>    def inner():
>        nonlocal x # nonlocal 关键字表示这里的 x 就是外部函数 outer 定义的变量 x
>        x = 'nonlocal'
>        print("inner:", x)
>```

#### [257. 二叉树的所有路径](https://leetcode-cn.com/problems/binary-tree-paths/)

```shell
# 所有根节点到叶子结点的路径
# 递归
def binaryTreePaths(self, root: TreeNode) -> List[str]:
    paths = []
    def helper(root, path):
        if root:
            path += str(root.val)
            if not root.left and not root.right:
                paths.append(path)
            else:
                path += '->'
                helper(root.left, path)
                helper(root.right, path)
    helper(root, '')
    return paths

# 迭代
def binaryTreePaths(self, root: TreeNode) -> List[str]:
    if not root:
        return []
    paths = []
    pathQueue = [(root, str(root.val))]
    while pathQueue:
        node, path = pathQueue.pop()
        if not node.left and not node.right:
            paths.append(path)
        else:
            if node.left:
                pathQueue.append((node.left, path + '->' + str(node.left.val)))
            if node.right:
                pathQueue.append((node.right, path + '->' + str(node.right.val)))
    return paths
        
```



#### [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

```python
# 任意结点（可以从父到子，也可以子到父）到叶子结点的路径和最大
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        # 解析：一个树（root, left, right）三个结点，
        # 经过root有三条路径，root->left, left->root->right, root->left 
        res = float('-inf') # 初始化为负无穷，并不是 0 
        def maxGain(node):
            if not node:
                return 0
            # 递归计算左右子节点的最大贡献值
            # 只有在最大贡献值大于 0 时，才会选取对应子节点
            leftGain = max(maxGain(node.left), 0)
            rightGain = max(maxGain(node.right), 0)
            
            # 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
            priceNewpath = node.val + leftGain + rightGain
            
            # 更新答案
            nonlocal res
            res = max(res, priceNewpath)
        
            # 返回节点的最大贡献值
            return node.val + max(leftGain, rightGain)
   
        maxGain(root)
        return res
```



#### [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

```shell
# 根结点到叶子结点的路径和是否等于target？
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:

        # 递归 
        if not root:
            return False
        if not root.left and not root.right:
            return root.val == targetSum
        return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)

        # 迭代
        if not root:
            return False
        stack = [(root, root.val)]
        while stack:
            node, path = stack.pop()
            if not node.left and not node.right and path == targetSum:
                return True 
            else:
                if node.left:
                    stack.append((node.left, path + node.left.val))
                if node.right:
                    stack.append((node.right, path + node.right.val))
        return False
```



#### [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

```shell
# 问题：# 给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。 
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        # 递归 
        # targetSum 表示还剩多少
        paths = []
        def helper(root, path, targetSum):
            if root:
                # path.append(root.val)  # 错误做法：先将结点假如到path中，然后再去判断，这样会导致路径和大于目标值
                if not root.left and not root.right and targetSum - root.val == 0:
                    paths.append(path + [root.val])
                else:
                    helper(root.left, path + [root.val], targetSum - root.val)
                    helper(root.right, path + [root.val], targetSum - root.val)
        helper(root, [], targetSum)
        return paths

        # 迭代
        if not root:
            return []
        stack = [(root, [], 0)] # 当前要处理结点A，还没处理结点A之前的路径，还没处理结点A之前路径和
        paths = []
        while stack:
            node, path, pathTotal = stack.pop() 
            if not node.left and not node.right:
                if pathTotal + node.val == targetSum:
                    paths.append(path + [node.val])
            else:
                if node.left:
                    stack.append((node.left, path+[node.val], pathTotal + node.val)) # 问题，append里面不能再写append
                if node.right:
                    stack.append((node.right, path+[node.val], pathTotal + node.val))
        return paths
```



#### [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

```shell
# 任意结点（从父到子）到任意结点的路径和等于target的数量
```



#### [129. 求根到叶子节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

```shell
# 根结点到叶子结点的路径代表一个数字，求所有路径的和
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        # 递归
        def dfs(root, preTotal):
            if not root:
                return 0
            total = preTotal * 10 + root.val
            if not root.left and not root.right:
                return total
            else:
                return dfs(root.left, total) + dfs(root.right, total)
        return dfs(root, 0)

        # 迭代
        if not root:
            return 0
        stack = [(root, root.val)] # 存储当前结点，当前路径经过的结点构成的数字
        total = 0
        while stack:
            node, path = stack.pop()
            if not node.left and not node.right:
                total += path
            else:
                if node.left:
                    stack.append((node.left, path*10 + node.left.val))
                if node.right:
                    stack.append((node.right, path*10 + node.right.val))
        return total
```



#### [988. 从叶结点开始的最小字符串](https://leetcode-cn.com/problems/smallest-string-starting-from-leaf/)

```shell
# 每个结点是一个英文字母，求叶子结点到根结点所有路径中，字典排序最小
```



#### [404. 左叶子之和](https://leetcode-cn.com/problems/sum-of-left-leaves/)

```shell
# 计算所有左叶子之和
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        # 递归
        def dfs(root, flag = False): # 初始化flag = False
            if not root: 
                return 0 # 先判断root是否为None，否则下面的判断语句会报错
            if not root.right and not root.left and flag:  # 前两个条件保证是否是叶子节点，flag保证是否是左孩子
                return root.val # 如果是左叶子结点就加上其值
            return dfs(root.left, True) + dfs(root.right, False)  # 递归root的左孩子并让flag = True, 右孩子flag = False
        return dfs(root) # 返回结果

        # 迭代
        if not root:
            return 0
        isLeafNode = lambda node: not node.left and not node.right
        stack = [root]
        res = 0
        while stack:
            node = stack.pop()
            if node.left:
                if isLeafNode(node.left):
                    res += node.left.val
                else:
                    stack.append(node.left)
            if node.right:
                if not isLeafNode(node.right):
                    stack.append(node.right)
        return res
```



#### [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

```shell
# 任意两结点（可以从父到子，也可以子到父）之间的产长度，最长者为直径
class Solution:
    def maxDepth(self, root):
        if not root:
            return 0 
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        if not root:
            return 0
        return self.maxDepth(root.left) + self.maxDepth(root.right)
```



#### [545. 二叉树的边界](https://leetcode-cn.com/problems/boundary-of-binary-tree/)

```shell
# 左边界+右边界+叶子结点
```

#### [111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

```python
# 根到叶子结点最短路径上的节点数
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        # 迭代
        if not root:
            return 0
        queue = [(root, 1)]
        while queue:
            node, depth = queue.pop(0)
            if not node.left and not node.right:
                return depth
            if node.left:
                queue.append((node.left, depth + 1))
            if node.right:
                queue.append((node.right, depth + 1))
        return 0 
```



#### [662. 二叉树最大宽度](https://leetcode-cn.com/problems/maximum-width-of-binary-tree/)

```shell
# 求二叉树最宽的层，有多宽（指的是左右两个结点差距多远，并不是结点的个数）
```

#### [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

```shell
# 根结点到叶子结点的最长路径 
class Solution:
    # 递归三要素：返回类型+参数，结束条件，单个循环
    def maxDepth(self, root: TreeNode) -> int:
        # 递归
        if not root:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left,right) + 1

        # 迭代
        if not root:
            return 0
        stack = [(1, root)]
        depth = 1
        while stack:
            curDepth, node = stack.pop()
            depth = max(curDepth, depth)
            if node.left:
                stack.append((curDepth + 1, node.left))
            if node.right:
                stack.append((curDepth + 1, node.right))
        return depth
```









### 二叉树（遍历）

> **递归三要素**：确定返回类型、参数，确定结束条件，定义单层逻辑
>
> **python知识点**：extend（扩展）指的是将数组拼接到原始数组之后；append（追加）指的是在原始数组添加一个元素
>
> **python知识点**： list、tuple、str 都是有序的，tuple和str是不可改变对象

#### [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]: 
        # 递归
        def nlr(root, res):
            if not root:
                return []
            res.append(root.val)
            nlr(root.left, res)
            nlr(root.right, res)
            return res
        res = nlr(root, [])
        return res

        # 迭代 使用栈：先加右结点，再加左结点
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
```



#### [94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # 递归
        res = []
        def LNR(root, res):
            if not root:
                return []
            LNR(root.left, res)
            res.append(root.val)
            LNR(root.right, res)
            return res
        res = LNR(root, res)
        return res
        
        # 迭代, 使用栈，先加左结点，一直加到叶子结点
        res = []
        stack = []
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                node = stack.pop()
                res.append(node.val)
                root = node.right
        return res

```



#### [145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        # 递归
        def lrn(root, res):
            if not root:
                return []
            lrn(root.left, res)
            lrn(root.right, res)
            res.append(root.val)
            return res
        res = lrn(root, [])
        return res

        # 迭代
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return res[::-1]
```



#### [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            cur = []
            nextStack = []
            for node in stack:
                cur.append(node.val)
                if node.left:
                    nextStack.append(node.left)
                if node.right:
                    nextStack.append(node.right)
            res.append(cur)
            stack = nextStack
        return res
```



#### [589. N叉树的前序遍历](https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/)

```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        # # 递归
        res = []
        def helper(root, res):
            if not root:
                return []
            res.append(root.val)
            for node in root.children: # 重点
                helper(node, res)
            return res
        helper(root, res)
        return res

        # 迭代
        res = []
        if not root:
            return []
        stack = [root]
        while stack:
            node = stack.pop()
            res.append(node.val)
            for child in node.children[::-1]: # 前序遍历：从右侧到左侧添加叶子结点
                stack.append(child)
            # stack.extend(node.children[::-1]) # 前序遍历：从右侧到左侧添加叶子结点
        return res
```



#### [429. N 叉树的层序遍历](https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/)

```python
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        res = []
        if not root:
            return []
        queue = [root]
        while queue:
            cur = []
            nextQueue = []
            for node in queue:
                cur.append(node.val)
                for child in node.children:
                    nextQueue.append(child)
            queue = nextQueue
            res.append(cur)
        return res
```



#### [590. N叉树的后序遍历](https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/)

```python
class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        # 递归
        res = []
        def helper(root, res):
            if not root:
                return []
            for child in root.children:
                helper(child, res)
            res.append(root.val)
            return res
        helper(root, res)
        return res

        # 迭代
        res = []
        if not root:
            return []
        stack = [root]
        while stack:
            node = stack.pop()
            res.append(node.val)
            # for child in node.children: # 后续遍历：从左到右添加叶子结点
            #     stack.append(child)
            stack.extend(node.children) # 后续遍历：从左到右添加叶子结点
        return res[::-1]
            
```



#### [559. N 叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-n-ary-tree/)

```python
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        # 递归：时间O(N)，空间：当树平衡时，最好O(log(N))，当每个树仅有一个孩子，最差为O(N)
        if not root:
            return 0
        elif root.children == []:
            return 1
        return 1 + max([self.maxDepth(node) for node in root.children])

        # 迭代： 时间O(N)，空间O(N)
        stack = []
        if root:
            stack.append((1,root)) # python list、tuple、str 都是有序的，tuple和str是不可改变对象
        depth = 0
        while stack:
            currentDepth, node = stack.pop()
            if node:
                depth = max(depth, currentDepth)
            for child in node.children:
                stack.append((currentDepth+1, child))
        return depth
```





### 二叉树（构造）

#### [297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ''
        res = []
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                res.append('None')
        return ','.join(res)
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return None
        vals = data.split(',')
        i = 1
        root = TreeNode(vals[0])
        queue = [root]

        while queue:
            node = queue.pop(0)
            if vals[i] != 'None':
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1 # 放在if外面
            if vals[i] != 'None':
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1 # 放在if外面
        return root
```



#### [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```python
# tips:没有重复元素
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder or not inorder:
            return 
        val = preorder[0]
        index = inorder.index(val)

        leftIn = inorder[:index]
        rightIn = inorder[index+1:]
        leftPre = preorder[1:index+1]
        rightPre = preorder[index+1:]
        
        # 构建树
        root = TreeNode(val)
        left = self.buildTree(leftPre, leftIn)
        right = self.buildTree(rightPre, rightIn)
        root.left = left
        root.right = right
        return root
```



#### [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if not inorder or not postorder:
            return 
        val = postorder[-1]
        index = inorder.index(val)

        leftIn = inorder[:index]
        rightIn = inorder[index+1:]
        leftPost = postorder[:index]
        rightPost = postorder[index:-1]

        left = self.buildTree(leftIn, leftPost)
        right = self.buildTree(rightIn, rightPost)
        root = TreeNode(val)
        root.left = left
        root.right = right

        return root
```



#### [654. 最大二叉树](https://leetcode-cn.com/problems/maximum-binary-tree/)

```python
# 问题：不含重复元素的整数数组，构建最大二叉树
# 最大二叉树递归定义：根结点是nums中最大元素，最大值左侧数字递归构建最大二叉树，右侧递归构建最大二叉树
# 时空：O(n^2)， O(n)
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        def buildMaxBinTree(nums):
            if not nums:
                return 
            maxNum = max(nums)
            index = nums.index(maxNum)
            left = buildMaxBinTree(nums[:index])
            right = buildMaxBinTree(nums[index+1:])
            maxNode = TreeNode(maxNum, left, right)
            return maxNode
        return buildMaxBinTree(nums)
```



#### [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

```python
# 时空：O(N)，O(N)
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return 
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        root.left, root.right = right, left
        return root

```



#### [617. 合并二叉树](https://leetcode-cn.com/problems/merge-two-binary-trees/)

```python
# 问题：覆盖两个二叉树，如果那个位置有两个结点，则值为两个结点之和
class Solution:
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        # 递归 O(min(m,n))， O(min(m,n))
        if not t1:
            return t2
        if not t2:
            return t1
        merged = TreeNode(t1.val + t2.val)
        merged.left = self.mergeTrees(t1.left, t2.left)
        merged.right = self.mergeTrees(t1.right, t2.right)
        return merged
    	
        # 迭代：O(min(m,n))， O(min(m,n))
        # 待写。。。。
```







### 二叉树（其他）

#### [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 递归，时空：O(N)，O(N)
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left:
            return right
        if not right:
            return left
        return root
```



#### [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

```python
# 问题：二叉树转换成一个链表（每个结点只有右孩子）
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return root
        self.flatten(root.left)
        self.flatten(root.right)

        leftChild = root.left
        rightChild = root.right

        root.left = None
        root.right = leftChild

        p = root
        while(p.right != None):
            p = p.right
        p.right = rightChild
```



#### [116. 填充每个节点的下一个右侧节点指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)

```python
# 问题：满二叉树，每个结点添加一个next指针，指向层序遍历相邻的右侧

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        
        if not root:
            return root
        
        # 从根节点开始
        leftmost = root
        
        while leftmost.left:
            
            # 遍历这一层节点组织成的链表，为下一层的节点更新 next 指针， 
            # 结点默认的next就是空的，右侧边界不用赋值
            head = leftmost
            while head:
                
                # 一个根两个左右，画三角形
                head.left.next = head.right
                
                # 两个根，画梯形
                if head.next:
                    head.right.next = head.next.left
                
                # 指针向后移动
                head = head.next
            
            # 去下一层的最左的节点
            leftmost = leftmost.left
        
        return root 
```



#### [117. 填充每个节点的下一个右侧节点指针 II](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node-ii/)

```python
# 问题：任意二叉树，每个结点添加一个next指针，指向层序遍历相邻的右侧
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return 
        queue = [root]
        res = []
        while queue:
            # 先把当前层的next指针连上
            if len(queue) == 1:
                queue[0].next = None
            else:
                cur = queue[0]
                for i in range(1, len(queue)):
                    cur.next = queue[i]
                    cur = queue[i]
                queue[-1].next = None     

            # 获取下一层
            cur = []
            nextQueue = []
            for one in queue:
                cur.append(one.val)
                if one.left:
                    nextQueue.append(one.left)
                if one.right:
                    nextQueue.append(one.right)
            queue = nextQueue
        return root
```



#### [222. 完全二叉树的节点个数](https://leetcode-cn.com/problems/count-complete-tree-nodes/)

```python
# 思路1 
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        # 递归 时间复杂度为O(n)，空间复杂度为O(1)【不考虑递归调用栈】
        if not root:
            return 0
        left = self.countNodes(root.left)
        right = self.countNodes(root.right)
        return 1 + left + right
# 思路2 时间复杂度：O(logn * logn)，空间复杂度：O(logn)

```

#### [【剑指offer】完全二叉树最后一层的最右节点](https://blog.csdn.net/zjwreal/article/details/96027056)

```python
# 问题，使用long
```



#### [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

```python
# 时空：O(n)，O(n)
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def height(root): # 如果以root为根的树是平衡的，返回其高度，否则返回-1
            if not root:
                return 0
            leftHeight = height(root.left)
            rightHeight = height(root.right)
            if leftHeight == -1 or rightHeight == -1 or abs(leftHeight - rightHeight) > 1:
                return -1
            else:
                return max(leftHeight, rightHeight) + 1
        return height(root) >= 0
```

#### [100. 相同的树](https://leetcode-cn.com/problems/same-tree/)

```python
class Solution:
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """    
        # p and q are both None
        if not p and not q:
            return True
        # one of p and q is None
        if not q or not p:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.right, q.right) and \
               self.isSameTree(p.left, q.left)
```



#### [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

```python
# 问题：B是A的子结构， 即 A中有出现和B相同的结构和节点值。
# 题解不是很理解，isSameTree
class Solution:
    def isSameTree(self, root1, root2):
        if not root2: # b都遍历完了，还没发现不一样的，说明那就一样了
            return True
        if not root1:  # 压根就没有a，当然不行
            return False
        if root1.val != root2.val: # a b 的值不相等，肯定不行
            return False
        return self.isSameTree(root1.left, root2.left) and self.isSameTree(root1.right, root2.right) 
        
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        if not A or not B:
            return False
        if self.isSameTree(A, B):
            return True
        return self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B)
```



#### [572. 另一个树的子树](https://leetcode-cn.com/problems/subtree-of-another-tree/)

   ```python
   # 问题：判断s中是否包含t相同结构，和相同结点值的子树
   ```

#### [652. 寻找重复的子树](https://leetcode-cn.com/problems/find-duplicate-subtrees/)

#### [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        # 递归 O(n)， O(n)
        if not root:
            return True
        def dfs(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            if left.val != right.val:
                return False
            return dfs(left.left,right.right) and dfs(left.right, right.left)
        return dfs(root.left, root.right)

```







### 单调栈、单调队列

> ```python
> # 单调栈
> def getGreaterElement(nums):
>     stack = []
>     res = [-1 for _ in range(len(nums))]
>     for i in range(len(nums) - 1, -1, -1):
>         while stack and stack[-1] <= nums[i]:
>             stack.pop()
>         res[i] = -1 if not stack else stack[-1]
>         stack.append(nums[i])
>     return res
> print(getGreaterElement(heights))
> 
> ```
>
> ```python
> # 单调队列（由大到小）
> def pushNum(n):
>  # 把元素n加入双端队列，小于n的元素被n踩扁
>  while queue and queue[-1] < n:
>      queue.pop()
>  queue.append(n)
> 
> def popNum(n):
>  # 把最左端的元素n（若存在）弹出双端队列
>  if queue and queue[0] == n:
>      queue.pop(0)
> 
> 
> ```
>
> 

#### [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

```python
# 问题：nums中，数量超过⌊ n/2 ⌋的元素 ，用时空O(n)O(1)的方法
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # 摩尔投票法 ，因为多元元素指的是超过⌊ n/2 ⌋ ，所以每次选两个不同的数进行删除，剩下的一定是最多的那个
        count = 1 
        curNum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == curNum:
                count += 1
            else:
                count -= 1
                if count == 0:
                    curNum = nums[i+1]
        return curNum
```



#### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # # 思路1 暴力解法， 超时
        n = len(height)
        res = 0
        for i in range(1, n-1):
            lMax = 0
            rMax = 0
            for j in range(i, n):
                rMax = max(rMax, height[j])
            for j in range(i, -1, -1):
                lMax = max(lMax, height[j])
            res += min(lMax, rMax) - height[i]
        return res
        
        # 单调栈
        ans = 0
        stack = []
        for i in range(len(height)):
            while stack and height[stack[-1]] < height[i]:
                cur = stack.pop()
                if stack:
                    left = stack[-1]
                    right = i
                    curHeight = min(height[right], height[left]) - height[cur]
                    ans += (right - left - 1) * curHeight
            stack.append(i)
        return ans 

        # 双指针
        l = 0
        r = len(height) - 1
        res = 0
        while l < r:
            minHeight = min(height[l], height[r])
            if minHeight == height[l]:
                l += 1
                while height[l] < minHeight:
                    res += minHeight - height[l]
                    l += 1
            else:
                r -= 1
                while height[r] < minHeight:
                    res += minHeight - height[r]
                    r -= 1
        return res
```



#### [496. 下一个更大元素 I](https://leetcode-cn.com/problems/next-greater-element-i/)

```python
# 注意理解题意：两个数组元素不同，nums1中的元素，在nums2中对应的位置，并不是相同下标
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        res = {}
        for i in range(len(nums2)-1, -1, -1):
            while stack and stack[-1] <= nums2[i]:
                stack.pop()
            if stack:
                res[nums2[i]] = stack[-1]  # 这样做的原因是因为两个数组没有重复元素
            stack.append(nums2[i])
        return [res.get(x, -1) for x in nums1]
```



#### [503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/)

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        stack = []
        res = [-1 for _ in range(len(nums))]
        i = len(nums) - 1
        k = 2*len(nums) # 重点
        while k:
            while stack and stack[-1] <= nums[i]:
                stack.pop()
            res[i] = stack[-1] if stack else -1
            stack.append(nums[i])
            k -= 1
            i = (i - 1) % len(nums) # 重点
        return res
```



#### [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

```python
# 单调栈
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        stack = []
        res = [0 for _ in range(len(T))]
        for i in range(len(T)-1, -1, -1):
            while stack and T[i] >= stack[-1][1]:
                stack.pop()
            res[i] = 0 if not stack else stack[-1][0] - i 
            stack.append([i, T[i]])
        return res
```



#### [1118. 一月有多少天](https://leetcode-cn.com/problems/number-of-days-in-a-month/)

```python
# 闰年：能被4 整除 并且能被100整除，但是不能被400整除
```



#### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

![](https://img-blog.csdnimg.cn/20200624150009357.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L29wZW5jdnpzZWZ2,size_16,color_FFFFFF,t_70)

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 双端队列，左侧大，右侧小
        def pushNum(one):
            # 把元素n加入双端队列，小于n的元素被n踩扁
            while queue and queue[-1] < one:
                queue.pop()
            queue.append(one)
        def popNum(one):
            # 把最左端的元素n（若存在）弹出双端队列
            if queue and queue[0] == one:
                queue.pop(0)
        
        queue = []
        res = []
        for i in range(len(nums)):
            # 双端队列没有完全进入列表，则把该元素加入队列
            if i -k + 1 < 0:
                pushNum(nums[i])
            else:
                pushNum(nums[i]) # 把该元素加入队列
                res.append(queue[0]) # 最左端为最大值
                popNum(nums[i-k+1]) # 删除位置为i-k+1的元素 （若已经被压扁了，则不用删了）
        return res
```



#### [402. 移掉K位数字](https://leetcode-cn.com/problems/remove-k-digits/)

```python
# 问题：非负整数 num，移除这个数中的 k 位数字，使得剩下的数字最小。
# 思路：同样单调栈
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        remain = len(num) - k 
        for digit in num:
            while k and stack and stack[-1] > digit:
                stack.pop()
                k -= 1
            stack.append(digit)
        return ''.join(stack[:remain]).lstrip('0') or '0'                                      
```

#### [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram_area.png)

```python
# 思路：单调栈
# 42. 接雨水是找每个柱子左右两边第一个大于该柱子高度的柱子，而本题是找每个柱子左右两边第一个小于该柱子的柱子。
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        res = 0
        stack = [] # 存放下标索引的
        newHeights = [0] + heights + [0] # 首尾各自+个0

        # 开始遍历
        for i in range(len(newHeights)):
            # 如果栈不为空且当前考察的元素值小于栈顶元素值，
            # 则表示以栈顶元素值为高的矩形面积可以确定
            while stack and newHeights[stack[-1]] > newHeights[i]:
                cur = stack.pop() # 弹出栈顶元素
                curHeight = newHeights[cur] # 获取栈顶元素对应的高
                leftIndex = stack[-1] # 栈顶元素弹出后，新的栈顶元素就是其左侧边界
                rightIndex = i # 右侧边界是当前考察的索引
                curWidth = rightIndex - leftIndex - 1 # 计算矩形宽度
                res = max(res, curWidth*curHeight) # 计算面积
            stack.append(i) # 当前考察索引入栈
        return res
```



#### [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

![](https://assets.leetcode.com/uploads/2020/09/14/maximal.jpg)

```python
# 思路：单调栈
class Solution:
    def largestRectangleArea(self, heights):
        res = 0
        stack = [] # 存放下标索引的
        newHeights = [0] + heights + [0] # 首尾各自+个0
        # 开始遍历
        for i in range(len(newHeights)):
            # 如果栈不为空且当前考察的元素值小于栈顶元素值，
            # 则表示以栈顶元素值为高的矩形面积可以确定
            while stack and newHeights[stack[-1]] > newHeights[i]:
                cur = stack.pop() # 弹出栈顶元素
                curHeight = newHeights[cur] # 获取栈顶元素对应的高
                leftIndex = stack[-1] # 栈顶元素弹出后，新的栈顶元素就是其左侧边界
                rightIndex = i # 右侧边界是当前考察的索引
                curWidth = rightIndex - leftIndex - 1 # 计算矩形宽度
                res = max(res, curWidth*curHeight) # 计算面积
            stack.append(i) # 当前考察索引入栈
        return res

    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        # step1 获取每行为底的数组
        heights = []
        for one in matrix:
            if not heights:
                heights.append(list(map(int, one)))
            else:
                for i in range(len(one)):
                    if one[i] == '1':
                        one[i] = 1
                        one[i] += heights[-1][i]
                    else:
                        one[i] = 0
                heights.append(one)
        
        # step2 获取每行为底的最大面积
        maxArea = 0
        for one in heights:
            maxArea = max(maxArea, self.largestRectangleArea(one))
        return maxArea
```





### 优先队列

> 优先队列：大小堆
>
> **python知识点：heap使用**
>
> - `heapq.heappush`(*heap*, *item*)
>
>   将 *item* 的值加入 *heap* 中，保持堆的不变性。
>
> - `heapq.heappop`(*heap*)
>
>   弹出并返回 *heap* 的最小的元素，保持堆的不变性。如果堆为空，抛出 [`IndexError`](https://docs.python.org/zh-cn/3/library/exceptions.html#IndexError) 。使用 `heap[0]` ，可以只访问最小的元素而不弹出它。
>
> - `heapq.heappushpop`(*heap*, *item*)：先放入堆，再弹出最小
>
>   将 *item* 放入堆中，然后弹出并返回 *heap* 的最小元素。该组合操作比先调用 [`heappush()`](https://docs.python.org/zh-cn/3/library/heapq.html#heapq.heappush) 再调用 [`heappop()`](https://docs.python.org/zh-cn/3/library/heapq.html#heapq.heappop) 运行起来更有效率。
>
> - `heapq.heapify`(*x*)
>
>   将list *x* 转换成堆，原地，线性时间内。
>
> - `heapq.heapreplace`(*heap*, *item*)：先弹出原来最小，再放入堆
>
>   弹出并返回 *heap* 中最小的一项，同时推入新的 *item*。 堆的大小不变。 如果堆为空则引发 [`IndexError`](https://docs.python.org/zh-cn/3/library/exceptions.html#IndexError)。

#### [264. 丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/)

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        nums = [1]
        i2, i3, i5 = 0, 0, 0
        for i in range(1, n):
            ugly = min(nums[i2]*2, nums[i3]*3, nums[i5]*5)
            nums.append(ugly)
            if ugly == nums[i2]*2:
                i2 += 1
            if ugly == nums[i3]*3:
                i3 += 1
            if ugly == nums[i5]*5:
                i5 += 1
        return nums[-1]
```



#### [263. 丑数](https://leetcode-cn.com/problems/ugly-number/)

```python
class Solution:
    def isUgly(self, num: int) -> bool:
        if num <= 0:
            return False
        while num != 1:
            if num % 2 == 0:
                num = num // 2
            elif num % 3 == 0:
                num = num // 3
            elif num % 5 == 0:
                num = num // 5
            else:
                return False
        return True
```

#### [295. 数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)

```python
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.maxHeap = []
        self.minHeap = []

    # 思路：因为python默认的堆是小顶堆，所以构建大顶堆的时候，取每个元素的负数，这样这样堆顶就是原来的最大值
    # 
    def addNum(self, num: int) -> None:
        if len(self.maxHeap) == len(self.minHeap):
            heapq.heappush(self.minHeap, -heapq.heappushpop(self.maxHeap, -num)) # heappushpop(list, x) :先加入x，然后再弹出
        else:
            heapq.heappush(self.maxHeap, -heapq.heappushpop(self.minHeap, num)) # 大顶堆，存的都是原来的相反数

    def findMedian(self) -> float:
        if len(self.minHeap) == len(self.maxHeap):
            return (-self.maxHeap[0] + self.minHeap[0]) / 2
        else:
            return self.minHeap[0]

```



#### [703. 数据流中的第 K 大元素](https://leetcode-cn.com/problems/kth-largest-element-in-a-stream/)

```python
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k 
        self.nums = nums
        heapq.heapify(self.nums)
        while len(self.nums) > k:
            heapq.heappop(self.nums)

    def add(self, val: int) -> int:
        if len(self.nums) < self.k:
            heapq.heappush(self.nums, val)
        elif self.nums[0] < val:
            heapq.heapreplace(self.nums, val) # 先弹出原来最小，再push进新的元素，
        return self.nums[0]
```



#### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 双端队列，左侧大，右侧小
        def pushNum(one):
            # 把元素n加入双端队列，小于n的元素被n踩扁
            while queue and queue[-1] < one:
                queue.pop()
            queue.append(one)
        def popNum(one):
            # 把最左端的元素n（若存在）弹出双端队列
            if queue and queue[0] == one:
                queue.pop(0)
        
        queue = []
        res = []
        for i in range(len(nums)):
            # 双端队列没有完全进入列表，则把该元素加入队列
            if i -k + 1 < 0:
                pushNum(nums[i])
            else:
                pushNum(nums[i]) # 把该元素加入队列
                res.append(queue[0]) # 最左端为最大值
                popNum(nums[i-k+1]) # 删除位置为i-k+1的元素 （若已经被压扁了，则不用删了）
        return res
```



#### [692. 前K个高频单词](https://leetcode-cn.com/problems/top-k-frequent-words/)

```python
# 问题：如果单词频率相同，按照字典顺序输出 
```





### 栈、队列

#### [232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

```python
# 通过
```



#### [225. 用队列实现栈](https://leetcode-cn.com/problems/implement-stack-using-queues/)

```python
# 通过
```



#### [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

```python
# s中可能包含(),[],{}三种括号，判断s是否合法
# 思路：使用栈，遇到左括号进栈
class Solution:
    def isValid(self, s: str) -> bool:
        res = []
        for i in s:
            if i == '(' or i == '[' or i == '{' or res == []:
                res.append(i)
            elif i == ')' and res[-1] == "(":
                res.pop()
            elif i == ']' and res[-1] == '[':
                res.pop()
            elif i == '}' and res[-1] == '{':
                res.pop()
            else:
                res.append(i)
        return res == []
```



#### [1047. 删除字符串中的所有相邻重复项](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)

```python
# 用栈
class Solution:
    def removeDuplicates(self, S: str) -> str:
        stack = []
        for i in range(len(S)):
            if stack and stack[-1] == S[i]:
                stack.pop()
            else:
                stack.append(S[i])
        return ''.join(stack)
```



#### [150. 逆波兰表达式求值](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

```python
class Solution:
    def evalRPN(self, tokens):
        stack = []
        for i in range(len(tokens)):
            if tokens[i] == "+" or tokens[i] == "-" or tokens[i] == "*" or tokens[i] == "/":
                b = stack.pop()
                a = stack.pop()
                if tokens[i] == "+":
                    stack.append(a + b)
                elif tokens[i] == "-":
                    stack.append(a - b)
                elif tokens[i] == "*":
                    stack.append(a * b)
                else:
                    stack.append(int(a / b))    #注意此处，由于python整除是向下取整，所以改用除法，然后用整形去掉小数, 用a//b 通过不了
            else:
                stack.append(int(tokens[i]))
        return stack.pop()
```



#### [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

```python
# python知识点：字典排序，sorted：返回列表，.sort()是对元数据进行修改
# 字典排序（按照key）：sorted(d.items(), key= lambda x:x[0])
# 字典排序（按照value）：sorted(d.items(), key= lambda x:x[1])
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # 方法1 使用字典 
        numsDict = {}
        for one in nums:
            numsDict[one] = numsDict.setdefault(one, 0) + 1
        temp = sorted(numsDict.items(), key = lambda x: x[1], reverse = True)
        res = []
        for i in range(k):
            res.append(temp[i][0])
        return res
        # 方法2 使用堆
```

#### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

```python
class Solution:
    def heapify(self, arr, i, l):
        left = 2*i+1
        right = 2*i+2
        largest = i
        if left < l and arr[left] > arr[largest]:
            largest = left
        if right < l and arr[right] > arr[largest]:
            largest = right
        if i != largest:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, largest, l)
    def buildMaxHeap(self, arr, l):
        for i in range((l-1-1)//2, -1, -1):
            self.heapify(arr, i, l)
    def maxHeapSort(self, arr, l, k):
        self.buildMaxHeap(arr, l)
        for i in range(l-1, l-k-1, -1): # 注意细节
            arr[0], arr[i] = arr[i], arr[0]
            self.heapify(arr, 0, i)
        return arr
    def findKthLargest(self, nums: List[int], k: int) -> int:
        l = len(nums)
        arr = self.maxHeapSort(nums, l, k)
        return arr[-k]
```



#### [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

```python
# 画图好理解： [尾stack1头][头stack2尾部] --》 模拟 queue【头...尾】
class CQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def appendTail(self, value: int) -> None:
        self.stack1.append(value)

    def deleteHead(self) -> int:
        if not self.stack1 and not self.stack2:
            return -1
        elif self.stack1 and not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
        elif self.stack2:
            return self.stack2.pop()
```



#### [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = [math.inf]

    def push(self, x: int) -> None:
        self.stack.append(x)
        self.min_stack.append(min(x, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```



#### [剑指 Offer 59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

#### [946. 验证栈序列](https://leetcode-cn.com/problems/validate-stack-sequences/)

```python
# class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        j = 0
        for i in range(len(pushed)):
            stack.append(pushed[i])
            while stack and stack[-1] == popped[j]:
                stack.pop()
                j += 1
        return stack == []
```





### 链表（反转、合并）

> **链表知识点**
>
> ```python
> fast = head
> while fast: # 最终fast是一个空结点
>     fast = fast.next
> 
> while fast.next: # 最终fast是链表最后一个结点
>     fast = fast.next
> ```
>
> 

#### [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 方法1 递归，时空 O(m+n)
        if l1 == None:
            return l2
        elif l2 == None:
            return l1
        elif l1.val  < l2.val:
            l1.next = self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1,l2.next)
            return l2

        # 方法2 迭代
        prehead = ListNode(-1)
        prev = prehead
        while l1 and l2:
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 if l1 else l2
        return prehead.next

```

```c++
class Solution{
public:
  ListNode* mergeTwoLists(ListNode* l1, ListNode* l2){
    if(l1 == nullptr){
      return l2;
    } else if (l2 == nullptr){
      return l1;
    } else if (l1->val < l2->val){
      l1->next = mergeTwoLists(l1->next, l2);
      return l1;
    } else{
      l2->next = mergeTwoLists(l1, l2->next);
      return l2;
    }
  }
}
```





#### [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

```python
# 归并排序
class Solution:

    # 合并两个有序链表
    def mergeTwoLists(self, l1, l2):
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l2.next, l1)
            return l2
    
    def mergeSort(self, lists, left, right):
        if left == right:
            return lists[left]
        mid = left + (right - left) // 2
        l1 = self.mergeSort(lists, left, mid)
        l2 = self.mergeSort(lists, mid+1, right)
        return self.mergeTwoLists(l1, l2)
        
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return None
        else:
            return self.mergeSort(lists, 0, len(lists) - 1)
```



#### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

```python
# 反转单链表
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # 迭代
        pre = None
        while head:
            temp = head.next
            head.next = pre
            pre = head
            head = temp
        head = pre
        return head

        # 递归
        if not head or not head.next:
            return head
        p = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return p
```



#### [92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

```python
# 反转位置m到n的链表， 1<=m<=n <len(linkList)
# 输入: 1->2->3->4->5->NULL, m = 2, n = 4
# 输出: 1->4->3->2->5->NULL
# 思路：1. 翻转整个链表，2. 翻转前n个结点，3. 翻转m到n的结点 ，用递归的方法
class Solution:
    def __init__(self):
        self.successor = None
    def reverseN(self, head, n):
        if n == 1:
            self.successor = head.next
            return head
        last = self.reverseN(head.next, n-1)
        head.next.next = head
        head.next = self.successor
        return last
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        if m == 1:
            return self.reverseN(head, n)
        head.next = self.reverseBetween(head.next, m-1, n-1)
        return head
```



#### [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

```python
# 输入：1->2->3->4->5->NULL , k = 3
# 输出：3->2->1->4->5->NULL
# 思路：递归 O(n)，O(1)
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        cur = head
        count = 0
        while cur and count < k:
            cur = cur.next
            count += 1
        if count == k:
            cur = self.reverseKGroup(cur, k)
            while count:
                tmp = head.next
                head.next = cur
                cur = head
                head = tmp
                count -= 1
            head = cur
        return head
```



#### [143. 重排链表](https://leetcode-cn.com/problems/reorder-list/)

```python
# 输入：1->2->3->4->5->NULL
# 输出：1->5->2->4->3->NULL
# 思路1：快慢指针，找到前半部分 和 后半部分；将后半部分翻转；然乎依次合并两个表
class Solution:
    # 1. 找到中点
    def middleNode(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    # 2. 翻转链表 
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        curr = head
        while curr:
            nextTemp = curr.next
            curr.next = prev
            prev = curr
            curr = nextTemp
        return prev

    # 3. 合并两个链表
    def mergeList(self, l1: ListNode, l2: ListNode):
        while l1 and l2:
            l1_tmp = l1.next
            l2_tmp = l2.next

            l1.next = l2
            l1 = l1_tmp

            l2.next = l1
            l2 = l2_tmp
    # 4. 主函数
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return
        
        mid = self.middleNode(head)
        l1 = head
        l2 = mid.next
        mid.next = None
        l2 = self.reverseList(l2)
        self.mergeList(l1, l2)
```



#### [147. 对链表进行插入排序](https://leetcode-cn.com/problems/insertion-sort-list/)

```python
# 思路：插入排序
# 时空：O(n2) O(1)
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        dummy = ListNode(0)
        dummy.next = head
        lastSorted = head # 维护 lastSorted 为链表的已排序部分的最后一个节点，初始时 lastSorted = head
        cur = head.next # 维护 curr 为待插入的元素，初始时 curr = head.next

        while cur:
            if lastSorted.val <= cur.val:
                lastSorted = lastSorted.next
            else:
                pre = dummy # 每次都是从头开始比较
                while pre.next.val <= cur.val:
                    pre = pre.next
                lastSorted.next = cur.next
                cur.next = pre.next
                pre.next = cur 
            cur = lastSorted.next
        return dummy.next

```



#### [148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

```python
# 思路：链表归并排序
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # step1 递归跳出条件
        if not head or not head.next:
            return head
        
        # step2 找到链表中点，拆分成左右两份
        slow = head
        fast = head.next # 注意是head.next ,画图
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next 
        mid = slow.next
        slow.next = None 

        # step3 递归调用，得到两个有序链表
        left = self.sortList(head) # 左侧已经排好序了
        right = self.sortList(mid) # 右侧已经排好序了

        # step4 合并两个有序链表
        newHead = ListNode(0)
        temp = newHead
        while left and right:
            if left.val < right.val:
                temp.next = left
                left = left.next
            else:
                temp.next = right
                right = right.next
            temp = temp.next
        temp.next = left if left else right
        return newHead.next
```



#### [328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/)

```python
# 输入: 1->2->3->4->5->NULL
# 输出: 1->3->5->2->4->NULL
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        # 思路 拆分成两个
        if not head or not head.next:
            return head   
    
        odd = head # 奇数
        p = odd
        even = head.next # 偶数
        q = even

        while q and q.next:
            p.next = q.next
            p = p.next
            q.next = p.next
            q = q.next
        p.next = even
        return odd
            
```



#### [61. 旋转链表](https://leetcode-cn.com/problems/rotate-list/)

```python
# 问题：每个结点向右侧移动k个位置
# 输入: 1->2->3->4->5->NULL, k = 2
# 输出: 4->5->1->2->3->NULL
# tips：k可能大于链表的长度，所以最终要对长度
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        
        if not head or not head.next:
            return head
        tail = head
        length = 1
        while tail.next: # tail 落到最后一个结点
            tail = tail.next
            length += 1
        tail.next = head  # 理解：head 指针指向啥，tail.next 就指向啥; tial.next 指向head所指向


        k = k % length
        for _ in range(length-k):
            tail = tail.next
        
        head = tail.next
        tail.next = None
        return head
```



#### [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

```python
# 输入: 1->2->3->4->5->NULL
# 输出: 2->1->4->3->5->NULL
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        # 递归
        if not head or not head.next:
            return head
        newHead = head.next
        head.next = self.swapPairs(newHead.next)
        newHead.next = head
        return newHead

        # 迭代
        dummy = ListNode() # 创建哑结点
        dummy.next = head
        tmp = dummy
        
        while tmp.next and tmp.next.next:
            p = tmp.next
            q = tmp.next.next
            tmp.next = q 
            p.next = q.next
            q.next = p 
            tmp = p 
        return dummy.next
```



#### [86. 分隔链表](https://leetcode-cn.com/problems/partition-list/)

   ```python
# 问题：小于 x 的节点都出现在 大于或等于 x 的节点之前
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # 思路 拆成两个链表

        first = ListNode(0)
        p = first
        second = ListNode(0)
        q = second

        while head:
            if head.val < x:
                p.next = head
                p = p.next
                head = head.next
            else:
                q.next = head
                q = q.next
                head = head.next

        q.next = None 
        p.next = second.next
        return first.next
   ```

   

### 链表（其他）

#### [876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

```python
# 链表基础 查找中点 
# 例子1：输入 [1,2,3,4,5,6]，输出 [4,5,6]
# 例子2：输入 [1,2,3,4,5]，输出 [3,4,5]
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        fast = head
        slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        return slow
   

```



#### [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

```python
# 倒数第k个结点
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        fast = head
        slow = head
        while k > 0:
            fast = fast.next
            k -= 1
        while fast:
            fast = fast.next
            slow = slow.next
        return slow

```



#### [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

```python
# 思路，两个指针分别从头开始走，走到末尾，直接跳到另一个头上，继续走，直到相遇
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        p = headA
        q = headB
        while p != q:
            p = p.next if p else headB
            q = q.next if q else headA
        return p
```



#### [138. 复制带随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

```python
# 思路 使用hash + 递归
class Solution:
    def __init__(self):
        self.visitedHash = {}
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return head
        if head in self.visitedHash:
            return self.visitedHash[head]
        node = Node(head.val, None, None)
        self.visitedHash[head] = node
        node.next = self.copyRandomList(head.next)
        node.random = self.copyRandomList(head.random)
        return node 
```



#### [剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

```python
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        dummy = TreeNode(0)
        dummy.next = head
        pre = dummy
        cur = head 
        while cur:
            if cur.val == val:
                pre.next = cur.next
                break
            else:
                pre = pre.next
                cur = cur.next
        return dummy.next
```



#### [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

```python
# 问题：重复元素只保留一次（链表有序）
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return None
        slow = head
        fast = head
        while fast: # 循环到最后，fast == None
            # 草稿画图最清晰
            if slow.val != fast.val:
                slow.next = fast
                slow = slow.next
                fast = fast.next
            else:
                fast = fast.next
        slow.next = fast # 也对
        slow.next = None # 也对
        return head
```



#### [82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

```python
# 问题：重复元素全部删除（链表有序）
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = TreeNode(0) # 创建哑结点
        dummy.next = head
        pre = dummy; cur = head
        while cur: # 当前节点存在 
            while cur.next and cur.val == cur.next.val: # 下一个节点存在，且与当前节点值重复
                cur = cur.next # 当前节点后移
            if pre.next == cur: # 前一个节点的后节点为当前节点，意味着当前节点未移动，且后一个节点不重复
                pre = pre.next # 前一个节点后移
            else: # 前一个节点的后节点不为当前节点，意味着当前节点移动，且后一个节点重复
                pre.next = cur.next 
            cur = cur.next
        return dummy.next
```



#### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

```python
# 问题两个链表相加，返回链表（链表是逆序存放数字）
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(0)
        pre = dummy 
        carry = 0
        while l1 or l2 or carry:
            n1 = l1.val if l1 else 0
            n2 = l2.val if l2 else 0 
            cur = ListNode((n1 + n2 + carry) % 10)
            carry = (n1 + n2 + carry) // 10
            pre.next = cur 
            pre = pre.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None 
        return dummy.next








# 扩展，链表是正序存放数字





```







### 旋转数组

#### [48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

```python
# 问题：n*n矩阵，顺时针旋转90度
# 方法1：水平翻转+对角线翻转 时空 O(n2), O(1)
# 水平翻转 ：matrix[row][col] = matrix[n-1-row][col]
# 对角线翻转：matrix[row][col] = matrix[col][row]
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        # 水平翻转
        for i in range(n//2):
            for j in range(n):
                # matrix[row][col] = matrix[n-1-row][col]
                matrix[i][j] , matrix[n-1-i][j] = matrix[n-1-i][j], matrix[i][j] # 注意是交换不是赋值
        # 对角线翻转
        for i in range(n):
            for j in range(i+1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```



#### [54. 螺旋  矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

```python
# 问题：m*n的矩阵，顺时针旋转，返回所有元素
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        while matrix:
            res += matrix.pop(0)
            # 矩阵转置：行变列
            matrix = list(zip(*matrix))[::-1] # 注意要取反
        return res
```



#### [59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)

```python
# 问题：n*n矩阵，顺时针生成1 到 n2 的矩阵
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        arr = [[0 for _ in range(n)] for _ in range(n)]
        num = 1 
        j = 0 
        while num <= n*n:
            for i in range(j, n-j):
                arr[j][i] = num
                num += 1
            for i in range(j+1, n-j):
                arr[i][n-j-1] = num
                num += 1
            for i in range(n-j-2, j-1, -1):
                arr[n-j-1][i] = num
                num += 1
            for i in range(n-j-2, j, -1):
                arr[i][j] = num
                num += 1
            j += 1
        return arr
```



#### [剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

```python
# 问题：m*n的矩阵，顺时针旋转，返回所有元素，同54题
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        while matrix:
            res += matrix.pop(0)
            # 矩阵转置：行变列
            matrix = list(zip(*matrix))[::-1] # 注意要取反
        return res
```



### 剑指offer（未归类）

#### [剑指 Offer 20. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)

#### [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

```python
# 同70.爬楼梯
class Solution:
    def numWays(self, n: int) -> int:
        # dp[i] 表示跳上第i个台阶共有多少种跳法
        # base case dp[0] = 1 dp[1] = 1
        dp = [1 for _ in range(n+1)]
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n] % 1000000007
```



#### [剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

```python
# 小顶堆
```

#### [剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

#### [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

#### [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

#### [剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

#### [剑指 Offer 67. 把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

#### [剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

#### [剑指 Offer 66. 构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

#### [剑指 Offer 64. 求1+2+…+n](https://leetcode-cn.com/problems/qiu-12n-lcof/)

#### [剑指 Offer 16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

#### [剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

#### [剑指 Offer 44. 数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)



### [Catalan数问题](https://blog.csdn.net/garrulousabyss/article/details/86619962)

![这里写图片描述](https://img-blog.csdn.net/20150628170417849)

### [概率题](https://www.nowcoder.com/discuss/346845?type=2)

1. 从一副52张扑克牌中随机抽两种，颜色相等的概率
2. 54张牌，分成6份，每份9张牌，大小王在一起的概率
3. 52张牌去掉大小王，分成26*2两堆，从其中一堆取4张牌为4个a的概率
4. 一枚硬币，扔了一亿次都是正面朝上，再扔一次反面朝上的概率是多少
5. 有8个箱子，现在有一封信，这封信放在这8个箱子中（任意一个）的概率为4/5,不放的概率为1/5（比如忘记了）,现在我打开1号箱子发现是空的，求下面7个箱子中含有这封信的概率为？
6. 已知N枚真硬币，M枚假硬币（两面都是国徽），R次重复采样都是国徽，问真硬币的概率？
7. 一对夫妻有2个孩子，求一个孩子是女孩的情况下，另一个孩子也是女孩的概率
8. 有种癌症，早期的治愈率为0.8，中期的治愈率为0.5，晚期的治愈率为0.2.若早期没治好就会转为中期，中期没治好就会变成晚期。现在有一个人被诊断为癌症早期，然后被治愈了，问他被误诊为癌症的概率是多少
9. 某城市发生了一起汽车撞人逃跑事件，该城市只有两种颜色的车，蓝20%绿80%，事发时现场有一个目击者，他指证是蓝车，但是根据专家在现场分析，当时那种条件能看正确的可能性是80%，那么，肇事的车是蓝车的概率是多少？
10. 100人坐飞机，第一个乘客在座位中随便选一个坐下，第100人正确坐到自己坐位的概率是？
11. 一个国家重男轻女，只要生了女孩就继续生，直到生出男孩为止，问这个国家的男女比例？
12. 有50个红球，50个蓝球，如何放入两个盒子中使得拿到红球的概率最大
13. 某个一直函数返回0/1，0的概率为p，写一函数返回两数概率相等
14. 给你一个函数，这个函数是能得出1-5之间的随机数的，概率相同。现在求1-7之间随机函数
15. X是一个以p的概率产生1,1-p的概率产生0的随机变量，利用X等概率生成1-n的数
16. 一个硬币，你如何获得2/3的概率
17. 怎么计算圆周率π的值(蒙特卡洛采样)
18. 网游中有个抽奖活动，抽中各星座的概率为10/200，20/200，。。。120/200.如何实现？
19. 给一个概率分布均匀的随机数发生器，给一串float型的数，希望通过这个随机数发生器实现对这串数进行随机采样，要求是如果其中的某个数值越大，那么它被采样到的概率也越大
20. 随机数生成算法，一个float数组相同元素的和除以整个数组的和做为抽取该元素的概率，实现按这种概率随机抽取数组中的元素的算法
21. 一本无数个字的书从前往后读，某个时间点突然暂停并返回之前读过的某个字，要求每个字返回的概率是一样的。
22. 一个有n*n个方格的棋盘，在里面放m个地雷，如何放保证在每个方格上放雷的概率相等。
23. 一根棍子折三段能组成三角形的概率
24. 一个圆上三个点形成钝角的概率是多少？假如两个点和圆心形成的圆心角已经是直角，那么第三个和这两个点形成钝角的概率是多少？
25. X，Y独立均服从（0,1）上的均匀分布，P{X^2+Y^2≤1}等于
26. 一个圆，在圆上随机取3个点，这3个点组成锐角三角形的概率。
27. 一个袋子里有100个黑球和100个白球，每次随机拿出两个球丢掉，如果丢掉的是不同颜色的球，则从其他地方补充一个黑球到袋子里，如果颜色相同，则补充一个白球到袋子里。问：最后一个球是黑球和白球的概率分别为多大？
28.  扔骰子，最多扔两次，第一次扔完可以自行决定要不要扔第二次，去最后一次扔色子的结果为准，求：尽可能得到最大点数的数学期望
29. 某大公司有这么一个规定：只要有一个员工过生日，当天所有员工全部放假一天。但在其余时候，所有员工都没有假期，必须正常上班。这个公司需要雇用多少员工，才能让公司一年内所有员工的总工作时间期望值最大？
30. 不存储数据流的前提下,从输入流中获得这 n 个等概率的随机数据
31.  某段公路上1小时有车通过的概率是0.96，半小时有车通过的概率是多少
32.  一个公交站在1分钟内有车经过概率是p，问3分钟内有车经过概率
33. 8支球队循环赛，前四名晋级。求晋级可能性
34. 一个活动，女生们手里都拿着长短不一的玫瑰花，无序地排成一排，一个男生从队头走到队尾，试图拿到尽可能长的玫瑰花，规则是:一旦他拿了一朵，后面就不能再拿了，如果错过了某朵花，就不能再回头，问最好的策略是什么?
35. 三个范围在0-1的数，和也在0-1的概率。
36.  11个球，1个特殊球，两个人无放回拿球，问第一个人取到特殊球的概率
37. 抛硬币，正面继续抛，反面不抛。问抛的次数的期望。
38. 抛的硬币直到连续出现两次正面为止，平均要扔多少次
39. 均匀分布如何生成正态分布
40. 2个轻的砝码，5个重的砝码和一个天平，几轮可以找到轻的砝码？



### [智力题](https://www.nowcoder.com/discuss/150434?type=2)

1. 有 25 匹马和 5 条赛道，赛马过程无法进行计时，只能知道相对快慢。问最少需要几场赛马可以知道前 3 名。
2. 给定两条绳子，每条绳子烧完正好一个小时，并且绳子是不均匀的。问要怎么准确测量 15 分钟。
3. 有 9 个球，其中 8 个球质量相同，有 1 个球比较重。要求用 2 次天平，找出比较重的那个球。
4. 有 20 瓶药丸，其中 19 瓶药丸质量相同为 1 克，剩下一瓶药丸质量为 1.1 克。瓶子中有无数个药丸。要求用一次天平找出药丸质量 1.1 克的药瓶。
5. 有两个杯子，容量分别为 5 升和 3 升，水的供应不断。问怎么用这两个杯子得到 4 升的水。
6. 一栋楼有 100 层，在第 N 层或者更高扔鸡蛋会破，而第 N 层往下则不会。给 2 个鸡蛋，求 N，要求最差的情况下扔鸡蛋的次数最少。



### [概率智力题合集](https://www.nowcoder.com/discuss/526897?type=all&order=time&pos=&page=3&channel=-1&source_id=search_all_nctrack)

