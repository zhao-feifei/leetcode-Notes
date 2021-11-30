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



# 二.链表

## 1.判断链表类

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



## 3.链表删除类

## 4.其他



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



## 2.最长系列

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



# 四.二叉树

# 五.动态规划

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



