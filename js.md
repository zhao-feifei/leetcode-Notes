# 一.数组

## 1. n数之和

#### 两数之和

```javascript
var twoSum = function (nums, target) {
  for (let i = 0; i < nums.length; i++) {
    let k = target - nums[i]
    for (let j = i + 1; j < nums.length; j++) {
      if (nums[j] == k) return [i, j]
    }
  }
}
```

#### 三数之和

```javascript
var threeSum = function(nums) {
    let res=[];
    const len=nums.length;
    if(nums==null||len<3) return res;
    nums.sort((a,b)=>a-b);
    for(let i=0;i<len;i++){
        if(nums[i]>0) break;
        if(i>0&&nums[i]==nums[i-1]) continue;
        let L=i+1;
        let R=len-1;
        while(L<R){
            const sum=nums[i]+nums[L]+nums[R];
            if(sum==0){
                res.push([nums[i],nums[L],nums[R]]);
                while(L<R&&nums[L]==nums[L+1]) L++;
                while(L<R&&nums[R]==nums[R-1]) R--;
                L++;
                R--;
            }
            else if(sum>0) R--;
            else if(sum<0) L++;
        }
    }
    return res;
};

```





#### 合并两个有序数组

```javascript
const merge = (nums1, m, nums2, n) => {
        let i = nums1.length - 1
        m--
        n--
        while (n >= 0) {
            if (nums1[m] > nums2[n]) {
                nums1[i--] = nums1[m--]
            } else {
                nums1[i--] = nums2[n--]
            }
        }
    
    
    // 当n = -1 的时候, 说明nums1[0] 的位置是正确的, 此时也就无需进行赋值, 结束循环即可

    // 当m = -1 的时候, nums2[n] 会得到一个undefined

    // undefined跟数字进行比较的时候, 会转换成NaN

    // NaN(Not a Number) 跟任何数字进行比较的时候, 都会得到一个 false

    // 也就意味着必定进入else, 那我们正好可以把nums2[0] 的值赋给nums1[0], 然后正好结束循环
}
```

## 2.最值系列

#### 最大子序和

```javascript
var maxSubArray = function (nums) {
    let ans=nums[0];
    let sum=0;
    for(let num of nums){
        if(sum>0){
            sum+=num;
        }else{
            sum=num;
        }
         ans = Math.max(sum, ans)
    }
    return ans;
};
```

#### 数组中的第K个最大元素

```javascript
const findKthLargest = (nums, k) => {
  const n = nums.length;

  const quick = (l, r) => {
    if (l > r) return;
    let random = Math.floor(Math.random() * (r - l + 1)) + l; // 随机选取一个index
    swap(nums, random, r); // 将它和位置r的元素交换，让 nums[r] 作为 pivot 元素
    /**
     * 我们选定一个 pivot 元素，根据它进行 partition
     * partition 找出一个位置：它左边的元素都比pivot小，右边的元素都比pivot大
     * 左边和右边的元素的是未排序的，但 pivotIndex 是确定下来的
    */
    let pivotIndex = partition(nums, l, r);
    /**
     * 我们希望这个 pivotIndex 正好是 n-k
     * 如果 n - k 小于 pivotIndex，则在 pivotIndex 的左边继续找
     * 如果 n - k 大于 pivotIndex，则在 pivotIndex 的右边继续找
     */
    if (n - k < pivotIndex) { 
      quick(l, pivotIndex - 1);
    } else {
      quick(pivotIndex + 1, r);
    }
    /**
     * n - k == pivotIndex ，此时 nums 数组被 n-k 分成两部分
     * 左边元素比 nums[n-k] 小，右边比 nums[n-k] 大，因此 nums[n-k] 就是第K大的元素
     */
  };

  quick(0, n - 1); // 让n-k位置的左边都比 nums[n-k] 小，右边都比 nums[n-k] 大
  return nums[n - k]; 
};

```

#### 长度最小的子数组

```javascript
var minSubArrayLen = function (target, nums) {
  let len = Infinity,
    i = 0,
    j = 0,
    sum = 0
  while (j < nums.length) {
    sum += nums[j]
    while (sum >= target) {
      len = Math.min(len, j - i + 1)
      sum -= nums[i]
      i++
    }
    j++
  }
  return len == Infinity ? 0 : len
}
```

#### 最大数

```javascript
var largestNumber = function(nums) {
    nums = nums.sort((a, b) => {
        let S1 = `${a}${b}`;
        let S2 = `${b}${a}`;
        return S2 - S1;
    });
    return nums[0] ? nums.join('') : '0';
};
```



## 3.矩阵

#### 螺旋矩阵

```javascript
var spiralOrder = function(matrix) {
    let res=[];
    flag=true;
    while(matrix.length){
        if(flag){
            //放进第一行   注意这里需要拼接而不是直接调用，否则覆盖之前的
            res=res.concat(matrix.shift());
            // 放进最后一列
            for(let i=0;i<matrix.length;i++){
                matrix[i].length&&res.push(matrix[i].pop());
            }
        }else{
            //最后一行
            res=res.concat(matrix.pop().reverse());
            //第一列
            for(let i=matrix.length-1;i>0;i--){
                matrix[i].length&&res.push(matrix[i].shift());
            }
        }
        flag=!flag;
    }
    return res;
};
```

#### 旋转图像

```javascript
var rotate = function (matrix) {
    const n = matrix.length;
    for (let i = 0; i < Math.floor(n / 2); i++) {
        for (let j = 0; j < n; j++) {
            [matrix[i][j], matrix[n - i - 1][j]] = [matrix[n - i - 1][j], matrix[i][j]];//注意这里是n-i-1
        }
    }
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < i; j++) {
            [matrix[i][j], matrix[j][i]] = [matrix[j][i], matrix[i][j]]
        }
    }
};
```



#### 岛屿数量

```
```

#### 岛屿最大面积

```
```

#### 二维数组中的查找

```javascript
var findNumberIn2DArray = function(matrix, target) {
    if(!matrix.length) return false;
    let x=matrix.length-1,y=0;
    while(x>=0&&y<matrix[0].length){
        if(matrix[x][y]==target){
            return true;
        }
        else if(matrix[x][y]>target){
            x--;
        }else{
            y++;
        }
    }
    return false;
};
```



## 4.数组中的查找

#### 二分查找

```javascript
var search = function (nums, target) {
  let start = 0
  end = nums.length - 1
  while (start <= end) {
    let mid = Math.floor((start + end) / 2)
    if (nums[mid] < target) {
      start = mid + 1
    } else if (nums[mid] > target) {
      end = mid - 1
    } else {
      return mid
    }
  }
  return -1
}
```

#### 寻找旋转排序数组中的最小值

```javascript
var findMin = function (nums) {
    let low=0,high=nums.length-1;
   
    while(low<high){
         let mid = low + Math.floor((high - low) / 2);
        if(nums[mid]<nums[high]){
            high=mid;
        }else{
            low=mid+1;
        }
    }
    return nums[low];
};
```

#### 在排序数组中查找元素的第一个和最后一个位置

```javascript
var searchRange = function(nums, target) {
    let L = 0,
        R = nums.length - 1;
    while (nums[L] < target) L++;
    while (nums[R] > target) R--;
    let res= R - L < 0 ? 0 : R - L + 1;
    if(res==0){
        return[-1,-1];
    }else{
        return[L,R];
    }
};
```

#### 多数元素

```javascript
var majorityElement = function (nums) {
    let majority = nums[0];
    let count = 1;
    for (let i = 1; i < nums.length; i++) {
        if (count == 0) {
            majority = nums[i];
        }
        if (majority == nums[i]) {
            count++;
        } else {
            count--;
        }
    }
    return majority;
};
```



## 5.栈和队列

#### 有效的括号

```javascript
var isValid = function (s) {
  if (s.length % 2 !== 0) {
    return false
  }
  let arr = []
  for (let i = 0; i < s.length; i++) {
    switch (s[i]) {
      case '(': {
        arr.push('(')
        break
      }
      case '{': {
        arr.push('{')
        break
      }
      case '[': {
        arr.push('[')
        break
      }
      case ']': {
        if (arr.pop() !== '[') {
          return false
        }
        break
      }
      case ')': {
        if (arr.pop() !== '(') {
          return false
        }
        break
      }
      case '}': {
        if (arr.pop() !== '{') {
          return false
        }
        break
      }
    }
  }
  return !arr.length
}
```

#### 最小栈

```javascript
var MinStack = function () {
    this.xStack=[];
    this.minStack=[Infinity];
};

/** 
 * @param {number} val
 * @return {void}
 */
MinStack.prototype.push = function (val) {
    this.xStack.push(val);
    this.minStack.push(Math.min(this.minStack[this.minStack.length-1],val));
};

/**
 * @return {void}
 */
MinStack.prototype.pop = function () { 
    this.xStack.pop();
    this.minStack.pop();
};

/**
 * @return {number}
 */
MinStack.prototype.top = function () {
    return this.xStack[this.xStack.length-1];
};

/**
 * @return {number}
 */
MinStack.prototype.getMin = function () {
    return this.minStack[this.minStack.length - 1]
};
```

#### 两个栈实现队列

```javascript
var CQueue = function () {
  this.stackA = []
  this.stackB = []
}

/**
 * @param {number} value
 * @return {void}
 */
CQueue.prototype.appendTail = function (value) {
  this.stackA.push(value)
}

/**
 * @return {number}
 */
CQueue.prototype.deleteHead = function () {
  if (this.stackB.length) {
    return this.stackB.pop()
  } else {
    while (this.stackA.length) {
      this.stackB.push(this.stackA.pop())
    }
    if (!this.stackB.length) {
      return -1
    } else {
      return this.stackB.pop()
    }
  }
}
```



## 6.其他

#### 合并区间

```javascript
var merge = function (intervals) {
  let res = []
  intervals.sort((a, b) => a[0] - b[0])
  let pre = intervals[0]
  for (let i = 1; i < intervals.length; i++) {
    let cur = intervals[i]
    if (pre[1] >= cur[0]) {
      pre[1] = Math.max(cur[1], pre[1])
    } else {
      res.push(pre)
      pre = cur
    }
  }
  res.push(pre)
  return res
}
```

#### 接雨水

```javascript
function trap(height) {
    let left = 0,
        right = height.length - 1,
        leftMax = 0,
        rightMax = 0,
        ans = 0;
    while (left < right) {
        leftMax = Math.max(leftMax, height[left]);
        rightMax = Math.max(rightMax, height[right]);
        if (leftMax < rightMax) {
            ans += leftMax - height[left];
            left++;
        } else {
            ans += rightMax - height[right];
            right--;
        }
    }
    return ans;
}
```

#### 两个数组的交集

```javascript
var intersection = function (nums1, nums2) {
  nums1.sort((a, b) => a - b)
  nums2.sort((a, b) => a - b)
  let res = new Set(),
    i = 0,
    j = 0
  while (i < nums1.length && j < nums2.length) {
    if (nums1[i] < nums2[j]) {
      i++
    } else if (nums1[i] > nums2[j]) {
      j++
    } else {
      res.add(nums1[i])
      i++
      j++
    }

  }
      return [...res]
}
```

#### 移动零

```javascript
var moveZeroes = function (nums) {
  if (nums == null) return
  let j = 0
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] != 0) {
      let temp = nums[i]
      nums[i] = nums[j]
      nums[j++] = temp
    }
  }
  return nums
}
```

#### 矩形重叠

```javascript
var isRectangleOverlap = function(rec1, rec2) {
    return !(rec1[2] <= rec2[0] ||   // left
             rec1[3] <= rec2[1] ||   // bottom
             rec2[2] <= rec1[0] ||   // right
             rec2[3] <= rec1[1]);    // top
}

```

#### 将数组分成和相等的三部分

```javascript
var canThreePartsEqualSum = function(A) {
    let sum = A.reduce((acc,cur)=>acc+cur) //sum数组之和
    let temp = 0   //temp累加
    let cnt = 0   //cnt计数
    for(let i=0;i<A.length;i++){
        temp += A[i] 
        if(temp == sum/3){  
            cnt++   
            temp = 0
        }
    }
    return (sum!=0 && cnt==3)||(sum==0 && cnt>2)
};

```

#### 下一个排列

```javascript
function nextPermutation(nums) {
    let i = nums.length - 2;                   // 向左遍历，i从倒数第二开始是为了nums[i+1]要存在
    while (i >= 0 && nums[i] >= nums[i + 1]) { // 寻找第一个小于右邻居的数
        i--;
    }
    if (i >= 0) {                             // 这个数在数组中存在，从它身后挑一个数，和它换
        let j = nums.length - 1;                // 从最后一项，向左遍历
        while (j >= 0 && nums[j] <= nums[i]) {  // 寻找第一个大于 nums[i] 的数
            j--;
        }
        [nums[i], nums[j]] = [nums[j], nums[i]]; // 两数交换，实现变大
    }
    // 如果 i = -1，说明是递减排列，如 3 2 1，没有下一排列，直接翻转为最小排列：1 2 3
    let l = i + 1;           
    let r = nums.length - 1;
    while (l < r) {                            // i 右边的数进行翻转，使得变大的幅度小一些
        [nums[l], nums[r]] = [nums[r], nums[l]];
        l++;
        r--;
    }
}

```



# 二.链表

## 1.判断链表类

#### 判断链表是否有环

```javascript
var hasCycle = function (head) {
  let fast = head
  let slow = head
  while (fast) {
    if (fast.next == null) return false
    fast = fast.next.next
    slow = slow.next
    if (fast == slow) {
      return true
    }
  }
  return false
}
```

#### 相交链表

```javascript
var getIntersectionNode = function(headA, headB) {
    if(!headA||!headB){
        return null
    }
    let PA=headA
    let PB=headB
    while(PA!=PB){
        PA=PA==null?headA:PA.next
         PB=PB==null?headB:PB.next
    }
    return PA
};
```



## 2.链表移动类

#### 反转链表

```javascript
var reverseList = function (head) {
  let pre = null,
    cur = head
  while (cur) {
    let next = cur.next
    cur.next = pre
    pre = cur
    cur = next
  }
  return pre
}
```

#### 合并两个有序链表

```javascript
var mergeTwoLists = function (l1, l2) {
  if (l1 == null) {
    return l2
  } else if (l2 == null) {
    return l1
  } else if (l1.val < l2.val) {
    l1.next = mergeTwoLists(l1.next, l2)
    return l1
  } else {
    l2.next = mergeTwoLists(l1, l2.next)
    return l2
  }
}
```

#### 两两交换链表中的节点

```javascript
var swapPairs = function(head) {
    if (head === null|| head.next === null) {
        return head;
    }
    const newHead = head.next;
    head.next = swapPairs(newHead.next);
    newHead.next = head;
    return newHead;
};

```

#### 回文链表

```javascript
//方法1：存入数组后判断数组是否为回文
var isPalindrome = function(head) {
        const vals=[];
        while(head!=null){
            vals.push(head.val);
            head=head.next;
        }
        for(var i=0,j=vals.length-1;i<j;i++,j--){
            if(vals[i]!=vals[j]){
                return false;
            }
        }
        return true;
};
//方法二：双指针
const isPalindrome = (head) => {
  if (head == null || head.next == null) {
    return true;
  }
  let fast = head;
  let slow = head;
  let prev;
  while (fast && fast.next) {
    prev = slow;
    slow = slow.next;
    fast = fast.next.next;
  }
  prev.next = null;  // 断成两个链表
  // 翻转后半段
  let head2 = null;
  while (slow) {
    const tmp = slow.next;
    slow.next = head2;
    head2 = slow;
    slow = tmp;
  }
  // 比对
  while (head && head2) {
    if (head.val != head2.val) {
      return false;
    }
    head = head.next;
    head2 = head2.next;
  }
  return true;
};

```



## 3.链表删除类

#### 删除链表的倒数第K个节点

```javascript
//另一道题，返回倒数第K个节点
var getKthFromEnd = function (head, k) {
  let p = head,
    q = head,
    i = 0
  while (p) {
    if (i >= k) {
      q = q.next
    }
    p = p.next
    i++
  }
  return i >= k ? q : null
}
```

#### 删除链表倒数第N个节点

```javascript
var removeNthFromEnd = function (head, n) {
  let ret = new ListNode(0, head),
    slow = (fast = ret)
  while (n--) fast = fast.next
  if (!fast) return ret.next
  while (fast.next) {
    fast = fast.next
    slow = slow.next
  }
  slow.next = slow.next.next
  return ret.next
}
```



## 4.其他

#### 两数相加

```javascript
var addTwoNumbers = function (l1, l2) {
  let head = null
  tail = null
  add = 0
  while (l1 || l2) {
    let n1 = l1 ? l1.val : 0
    let n2 = l2 ? l2.val : 0
    let sum = n1 + n2 + add
    if (!head) {
      head = tail = new ListNode(sum % 10)
    } else {
      tail.next = new ListNode(sum % 10)
      tail = tail.next
    }
    add = Math.floor(sum / 10)
    if (l1) {
      l1 = l1.next
    }
    if (l2) {
      l2 = l2.next
    }
    if (add > 0) {
      tail.next = new ListNode(add)
    }
  }
  return head
}
```



# 三.字符串

## 1.字符串处理

#### 字符串相加

```javascript
var addStrings = function (num1, num2) {
    let i=num1.length-1,
        j=num2.length-1;
    let add=0,
    ans=[];
    while(i>=0||j>=0||add!=0){
        let c1=i>=0?num1[i]-0:0,
            c2=j>=0?num2[j]-0:0;
        let sum=c1+c2+add;
        ans.push(sum%10);
        add=Math.floor(sum/10);
        i--;
        j--;

    }
    return ans.reverse().join('');
};
```

#### 比较版本号

```javascript
var compareVersion = function (version1, version2) {
  let arr1 = version1.split('.'),
    arr2 = version2.split('.')
  let len = Math.max(version1.length, version2.length)
  for (let i = 0; i < len; i++) {
    let data1 = 0,
      data2 = 0
    if (i < arr1.length) {
      data1 = parseInt(arr1[i])
    }
    if (i < arr2.length) {
      data2 = parseInt(arr2[i])
    }
    if (data1 < data2) {
      return -1
    }
    if (data1 > data2) {
      return 1
    }
  }
  return 0
}
```

#### 字符串相乘

```javascript
const multiply = (num1, num2) => {
  const len1 = num1.length;
  const len2 = num2.length;
  const pos = new Array(len1 + len2).fill(0);

  for (let i = len1 - 1; i >= 0; i--) {
    const n1 = +num1[i];
    for (let j = len2 - 1; j >= 0; j--) {
      const n2 = +num2[j];
      const multi = n1 * n2;             
      const sum = pos[i + j + 1] + multi; 

      pos[i + j + 1] = sum % 10;
      pos[i + j] += sum / 10 | 0;
    }
  }
  while (pos[0] == 0) {
    pos.shift();
  }
  return pos.length ? pos.join('') : '0';
};

```

#### 字符串解码

```javascript
var decodeString = function (s) {
  let num = 0
  result = ''
  numStack = []
  strStack = []
  for (const char of s) {
    if (!isNaN(char)) {
      num = num * 10 + Number(char)
    } else if (char == '[') {
      numStack.push(num)
      num = 0
      strStack.push(result)
      result = ''
    } else if (char == ']') {
      let repeatTime = numStack.pop()
      result = strStack.pop() + result.repeat(repeatTime)
    } else {
      result += char
    }
  }
  return result
}
```

#### 回文数

```javascript
var isPalindrome = function(x) {
    let str = x.toString();
    let n = str.length;
    let left = 0;
    let right = n-1;
    while(left < right){
        if(str[left++] != str[right--]){
            return false;
        }
    }
    return true;
}

```

#### 有效的字母异位词

```javascript
var isAnagram = function(s, t) {
    return s.split('').sort().join('') === t.split('').sort().join('')
};
//Sort不传参时，默认将元素转字符串，按每一个字节的Unicode编码位置值原地排序
//字符串 → 数组 → 排序 → 比较顺序相同的字符串
```



## 2.最长系列

#### 无重复字符的最长字串

```javascript
var lengthOfLongestSubstring = function (s) {
  let max = 0
  let ans = []
  for (let i = 0; i < s.length; i++) {
    let index = ans.indexOf(s[i])
    if (index != -1) {
      ans.splice(0, index + 1)
    }
    ans.push(s[i])
    max = Math.max(max, ans.length)
  }
  return max
}
```

#### 最长回文子串

```javascript
 var longestPalindrome = function (s) {
       if (s.length < 2) {
           return s
       }
       let res = ''
       for (let i = 0; i < s.length; i++) {
           // 回文子串长度是奇数
           helper(i, i)
           // 回文子串长度是偶数
           helper(i, i + 1)
       }

       function helper(m, n) {
           while (m >= 0 && n < s.length && s[m] == s[n]) {
               m--
               n++
           }
           // 注意此处m,n的值循环完后  是恰好不满足循环条件的时刻
           // 此时m到n的距离为n-m+1，但是mn两个边界不能取 所以应该取m+1到n-1的区间  长度是n-m-1
           if (n - m - 1 > res.length) {
               // slice也要取[m+1,n-1]这个区间 
               res = s.slice(m + 1, n)
           }
       }
       return res
   };
```

#### 最长公共子序列（不用连续）

```javascript
var longestCommonSubsequence = function (text1, text2) {
  let m = text1.length,
    n = text2.length
  let dp = Array.from(new Array(m + 1), () => new Array(n + 1).fill(0))
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      let c1 = text1[i - 1],
        c2 = text2[j - 1]
      if (c1 == c2) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
  }
  return dp[m][n]
}
```

#### 最长公共前缀

```javascript
var longestCommonPrefix = function (strs) {
  if (!strs.length) return ''
  let res = strs[0]
  for (let i = 1; i < strs.length; i++) {
    let j = 0
    for (; j < res.length && j < strs[i].length; j++) {
      if (strs[i][j] !== res[j]) {
        break
      }
    }
    res = res.substr(0, j)
  }
  return res
}
```



## 3.翻转系列

#### 翻转字符串里的单词

```javascript
var reverseWords = function (s) {

    let r = s.length - 1, l = r, res = "";
    while (l >= 0) {
        //先找到单词的尾部
        while (s[r] === " ") {
            r--;
        }
        l = r;

        //给上次单词加空格，排除第一次
        if (l >= 0 && res) {
            res += " ";
        }

        //再找到单词的头部
        while (s[l] && s[l] !== " ") {
            l--;
        }

        //遍历单词并添加
        for (let i = l + 1, j = r; i <= j; i++) {
            res += s[i];
        }

        //跳到下一个单词
        r = l;
    }

    return res;
};

```



# 四.二叉树

## 1.二叉树遍历

#### 前序遍历

```javascript
var preorderTraversal = function (root) {
  let res = []
  function preOrderTraversal(node) {
    if (!node) {
      return
    }
    res.push(node.val)
    preOrderTraversal(node.left)
    preOrderTraversal(node.right)
  }
  preOrderTraversal(root)
  return res
}

```



#### 中序遍历

```javascript
var inorderTraversal = function (root) {
  let res = []
  const insOrder = (root) => {
    if (!root) return
    insOrder(root.left)
    res.push(root.val)
    insOrder(root.right)
  }
  insOrder(root)
  return res
}
```

#### 二叉搜索树的第K小元素

```javascript
var kthSmallest = function (root, k) {
  let res
  function findNode(node) {
    if (node != null && k > 0) {
      findNode(node.left)
      if (--k == 0) {
        res = node.val
        return res
      }
      findNode(node.right)
    }
  }
  findNode(root)
  return res
}

```

#### 二叉搜索树的第K大元素

```javascript
var kthLargest = function (root, k) {
  let result =null,
  num = 0;
  function dfs(node){
    if(!node) return
    dfs(node.right)
    num++;
    if(num==k){
      result=node.val
      return 
    }
    dfs(node.left)
  }
  dfs(root)
  return result
}

```



## 2.二叉树路径

#### 路径总和

```javascript
var hasPathSum = function (root, targetSum) {
  if (!root) {
    return false
  }
  if (!root.left && !root.right) {
    return targetSum - root.val == 0
  }
  return hasPathSum(root.left,targetSum-root.val) || hasPathSum(root.right,targetSum-root.val)
}
```

#### 路径总和2

```javascript
var pathSum = function (root, targetSum) {
    let res=[];
    const dfs=(node,path,sum)=>{
        if(!node) return;
        path.push(node.val);
        sum+=node.val;
        if(!node.left&&!node.right){
            if(sum==targetSum){
                  res.push(path.slice());
            }
          
        }else{
            dfs(node.left,path,sum);
            dfs(node.right, path, sum);
        }
        path.pop();
    }
```



#### 求根节点到叶节点数字之和

```javascript
var sumNumbers = function (root) {
  const dfs = (root, preNum) => {
    if (!root) return 0
    let sum = preNum * 10 + root.val
    if (!root.left && !root.right) {
      return sum
    } else {
      return dfs(root.left, sum) + dfs(root.right, sum)
    }
  }

  return dfs(root, 0)
}
```

## 3.二叉树层序遍历

#### 层序遍历

```javascript
var levelOrder = function (root) {
  if (!root) return []
  let queue = [root]
  let res = []
  while (queue.length > 0) {
    let arr = []
    let len = queue.length
    while (len--) {
      let node = queue.shift()
      arr.push(node.val)
      if (node.left) queue.push(node.left)
      if (node.right) queue.push(node.right)
    }
    res.push(arr)
  }
  return res
}
```

#### N叉树层序遍历

```javascript
var levelOrder = function (root) {
    let nums=[];
    search(nums,root,0);
    return nums;
};
function search(nums,node,k){
    if(node==null){
        return;
    }
    if(nums[k]==undefined){
        nums[k]=[];
    }
    nums[k].push(node.val)
    for(let i=0;i<node.children.length;i++){
        search(nums,node.children[i],k+1)
    }
}
```



#### 二叉树锯齿形层序遍历

```javascript
var zigzagLevelOrder = function (root) {
    const res=[];
    if(!root){
        return res;
    }
    let curLevel=[root];
    while (curLevel.length){
        let curLevelVal=[];
        let nextLevel=[];
        for(const node of curLevel){
            curLevelVal.push(node.val);
            node.left && nextLevel.push(node.left);
            node.right && nextLevel.push(node.right);
        }
        res.push(curLevelVal);
         res.length % 2 == 0 && curLevelVal.reverse();//注意这里为啥放在后面
 
        curLevel=nextLevel;
    }
    return res;
};

```



## 4.二叉树的递归

#### 二叉树的最大深度

```javascript
var maxDepth = function (root) {
  if (!root) {
    return 0
  } else {
    let left = maxDepth(root.left)
    let right = maxDepth(root.right)
    return Math.max(left, right) + 1
  }
}
```

#### 通过前序与中序遍历序列构造二叉树

```javascript
var buildTree = function (preorder, inorder) {
     if(preorder.length==0) return null;
     let root=new TreeNode(preorder[0]);
     let mid=inorder.indexOf(preorder[0]);
      root.left = buildTree(preorder.slice(1, mid + 1), inorder.slice(0, mid));
     root.right = buildTree(preorder.slice(mid + 1), inorder.slice(mid + 1));
     return root;
 };
```

#### 二叉树的右视图

```javascript
var rightSideView = function (root) {
  let res = []
  if (!root) {
    return []
  }
  function dfs(node, step, arr) {
    if (node) {
      if (arr.length == step) {
        arr.push(node.val)
      }
      dfs(node.right, step + 1, arr)
      dfs(node.left, step + 1, arr)
    }
  }
  dfs(root, 0, res)
  return res
}

```

#### 翻转二叉树

```javascript
var invertTree = function (root) {
  if (!root) {
    return null
  }
  const right = invertTree(root.right)
  const left = invertTree(root.left)
  root.right = left
  root.left = right
  return root
}
```

#### 平衡二叉树

```javascript
var isBalanced = function (root) {
  function isBalanced(node) {
    if (!node) return 0
    let left = isBalanced(node.left)
    let right = isBalanced(node.right)
    if (left == -1 || right == -1 || Math.abs(left - right) > 1) {
      return -1
    }
    return Math.max(left, right) + 1
  }
  return isBalanced(root) !== -1
}
```

#### 二叉树的最近公共祖先

```javascript
const lowestCommonAncestor = (root, p, q) => {
    if (!root) return null;
    if (root == p || root == q) {
        return root;
    }
    const left = lowestCommonAncestor(root.left, p, q);
    const right = lowestCommonAncestor(root.right, p, q);
    if (left && right){
        return root;
    }
    if(left==null){
        return right;
    }
    return left;
};
```

#### 二叉搜索树中的搜索

```javascript
var searchBST = function (root, val) {
    if (root == null) return root
    if (root.val == val) return root
   return root.val < val ?  searchBST(root.right, val): searchBST(root.left, val)
};
```



# 五.动态规划

## 1.子序问题

#### 最长上升子序列

```javascript
var lengthOfLIS = function (nums) {
  let len = nums.length
  if (!len) return 0
  let dp = new Array(len).fill(1)
  for (let i = 1; i < len; i++) {
    for (let j = 0; j < i; j++) {
      if (nums[i] > nums[j]) {
        dp[i] = Math.max(dp[i], dp[j] + 1)
      }
    }
  }
  return Math.max(...dp)
}
```

#### 最长重复子数组

```javascript
var findLength = function (nums1, nums2) {
  let m = nums1.length
  let n = nums2.length
  let res = 0
  let dp =  Array.from(new Array(m + 1), () => new Array(n + 1).fill(0))
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (nums1[i - 1] == nums2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      }
      res = Math.max(res, dp[i][j])
    }
  }
  return res
}
```



#### 爬楼梯

```javascript
var climbStairs = function (n) {
  let dp = []
  dp[1] = 1
  dp[2] = 2
  for (let i = 3; i <= n; i++) {
    dp[i] = dp[i - 1] + dp[i - 2]
  }
  return dp[n]
}
```

## 2.金钱问题

#### 买卖股票的最佳时机

```javascript
var maxProfit = function (prices) {
  let min = prices[0]
  max = 0
  for (let i = 1; i < prices.length; i++) {
    if (prices[i] < min) {
      min = prices[i]
      continue
    }
    max = Math.max(max, prices[i] - min)
  }
  return max
}
```

#### 买卖股票的最佳时机 ||

```javascript
var maxProfit = function(prices) {
    let max=0,n=prices.length;
    for(let i=0;i<n-1;i++){
        max+=Math.max(0,prices[i+1]-prices[i])
    }   
    return max;
};
```



#### 零钱兑换

```javascript
var coinChange = function (coins, amount) {
  let dp = new Array(amount+1).fill(Infinity)
  dp[0] = 0
  for (let i = 1; i <= amount; i++) {
    for (let coin of coins) {
      if (coin <= i) {
        dp[i] = Math.min(dp[i], dp[i - coin] + 1)
      }
    }
  }
  return dp[amount] == Infinity ? -1 : dp[amount]
}
```

#### 打家劫舍

```javascript
var rob = function (nums) {
  let n = nums.length
  let dp = new Array(n + 1)
  dp[0] = 0
  dp[1] = nums[0]
  for (let i = 2; i <= n; i++) {
    dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1])
  }
  return dp[n]
}

```



## 3.路径问题

#### 最小路径和

```javascript
var minPathSum = function (grid) {
    let m=grid.length;
    let n=grid[0].length;
    let dp=Array.from(new Array(m),()=>Array(n).fill(0));
    dp[0][0]=grid[0][0];
    for(let i=1;i<m;i++){
        dp[i][0]=dp[i-1][0]+grid[i][0];
    }
    for (let j = 1; j < n; j++) {
        dp[0][j] = dp[0][j-1] + grid[0][j];
    }
    for(let i=1;i<m;i++){
        for(let j=1;j<n;j++){
            dp[i][j]=Math.min(dp[i-1][j],dp[i][j-1])+grid[i][j];
        }
    }
    return dp[m-1][n-1];
};
```

#### 不同路径

```javascript
var uniquePaths = function (m, n) {
  let dp = Array.from(new Array(m), () => new Array(n).fill(0))
  for (let i = 0; i < m; i++) {
    dp[i][0] = 1
  }
  for (let i = 0; i < n; i++) {
    dp[0][i] = 1
  }
  for (let i = 1; i < m; i++) {
    for (let j = 1; j < n; j++) {
      dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    }
  }
  return dp[m - 1][n - 1]
}
```



# 六.回溯算法

## 1.数组

#### 全排列

```javascript
var permute = function (nums) {
  let res = [],
    used = {}
  function dfs(path) {
    if (path.length == nums.length) {
      res.push(path.slice())
      return
    }
    for (const num of nums) {
      if (used[num]) {
        continue
      }
      path.push(num)
      used[num] = true
      dfs(path)
      path.pop()
      used[num] = false
    }
  }
  dfs([])
  return res
}
```

#### 子集

```javascript
var subsets = function (nums) {
    let res=[];
    const dfs=(index,list)=>{
        if(index==nums.length){
            res.push(list.slice());
            return;
        }
        list.push(nums[index]);
        dfs(index+1,list);
        list.pop();
        dfs(index+1,list);

    }
    dfs(0,[]);
    return res;

};
```



## 2.字符串

#### 括号生成

```javascript
var generateParenthesis = function (n) {
  let res = []
  function dfs(l, r, str) {
    if (str.length == 2 * n) {
      res.push(str)
      return
    }
    if (l > 0) {
      dfs(l - 1, r, str + '(')
    }
    if (r > l) {
      dfs(l, r - 1, str + ')')
    }
  }
  dfs(n, n, '')
  return res
}
```

#### 复原IP地址

```javascript
 var restoreIpAddresses = function (s) {
     let res = [];
     const dfs = (subRes, start) => {
         if (subRes.length == 4 && start == s.length) {
             res.push(subRes.join('.'));
             return;
         }
         if (subRes.length == 4 && start < s.length) return;
         for (let len = 1; len <= 3; len++) {
             if (len != 1 && s[start] == '0') return;
             if (start + len > s.length) return;
             const str = s.substring(start, start + len);
             if (len == 3 && +str > 255) return;
             subRes.push(str);
             dfs(subRes, start + len);
             subRes.pop();
         }
     }
     dfs([], 0);
     return res;
 };
```



# 七.数学

#### 斐波那契数列

```javascript
var fib = function (n) {
  let n1 = 0,
    n2 = 1,
    sum = 0
  for (let i = 0; i < n; i++) {
    sum = (n1 + n2) % 1000000007
    n1 = n2
    n2 = sum
  }
  return n1
}
```

#### 圆圈中最后剩下的数字

```javascript
var lastRemaining = function(n, m) {
    let ans=0 //f（1）=0，只有一个
    for(let i =2;i<=n;i++) {//
        ans=(ans+m)%i;
    }
    return ans;
};
```



