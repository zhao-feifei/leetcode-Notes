## 容易

#### 415.字符串 相加

``` javascript
var addStrings = function (num1, num2) {
    let i = num1.length - 1,
        j = num2.length - 1,
        add = 0;
    let arr = [];
    while (i >= 0 || j >= 0 || add != 0) {
        const x = i >= 0 ? num1.charAt(i) - '0' : 0;// 这里减去字符0是与阿斯克码相减
        const y = j >= 0 ? num2.charAt(j) - '0' : 0;
        let result = x + y + add;
        arr.push(result % 10);
        add = Math.floor(result / 10);
        i -= 1;
        j -= 1;
    }
    return arr.reverse().join('');
};
```

#### 88.合并两个有序数组

``` javascript
var merge = function(nums1, m, nums2, n) {
    let index1 =m- 1,
        index2 =n - 1,
        tail =m+n-1;
        while(index1>=0&&index2>=0){
            if(nums1[index1]>nums2[index2]){
                nums1[tail]=nums1[index1];
                index1--;
                tail--;
            }else{
                nums1[tail]=nums2[index2];
                index2--;
                tail--;
            }
        }
        while(index1>=0&&tail>=0){
            nums1[tail]=nums1[index1];
            index1--;
            tail--;
        }
        while(index2>=0&&tail>=0){
            nums1[tail]=nums2[index2];
            index2--;
            tail--;
        }
        return nums1;
};
```

#### 1.两数之和

``` javascript
var twoSum = function(nums, target) {
    let start=0,end=nums.length-1;
    nums.sort();
    while (start!=end) {
        if(nums[start]+nums[end]==target){
         return [start,end];
        }
        if(nums[start]+nums[end]>target){
            end--;
        }
        if(nums[start]+nums[end]<target){
            start++;
        }
    }
};
```

#### 70.爬楼梯

``` javascript
var climbStairs = function(n) {
    let dp=[];
    dp[1]=1;
    dp[2]=2;
    for (let i = 3; i <= n; i++) {
        dp[i]=dp[i-1]+dp[i-2];
        
    }
    return dp[n];
};
```

#### 206.反转链表

#### 112.路径总 和

#### 704.二分查找

```javascript
var search = function (nums, target) {
        let start = 0;
        let end = nums.length - 1;
        while (start <= end) {
            let mid = Math.floor((start + end) / 2);
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                end=mid-1;
            } else {
                start=mid+1;
            }
        };
        return -1;
    }

```

#### 20.有效的括号

```javascript
var isValid = function(s) {
    if (s.length % 2) {
        return false
    }
    let arr = [];
    for (let i = 0; i < s.length; i++) {
        letter = s[i];
        switch (letter) {
            case "(":
                {
                    arr.push(letter);
                    break;
                }
            case "{":
                {
                    arr.push(letter);
                    break;
                }
            case "[":
                {
                    arr.push(letter);
                    break;
                }
            case ")":
                {
                    if (arr.pop() !== "(") {
                        return false;
                    }
                    break;
                }
            case "}":
                {
                    if (arr.pop() !== "{") {
                        return false;
                    }
                    break;
                }
            case "]":
                {
                    if (arr.pop() !== "[") {
                        return false;
                    }
                    break;
                }
        }

    }
    return !arr.length;
}
```

#### 104.二叉树的最大深度

```javascript
var maxDepth = function(root) {
    if(!root){
        return 0;
    }else{
        const left=maxDepth(root.left);
        const right=maxDepth(root.right)
        return Math.max(left,right)+1;
    }
    
};
```

#### 141.环形链表

```javascript
var hasCycle=function(head){
    let fast=head;
    let slow=head;
    while(fast){
        if(fast.next==null){
            return false;
        }
            slow=slow.next;
            fast=fast.next.next;
            if(fast==slow){
                return true;
        }
    }
    return false;
}
```

#### 剑指offer  链表中倒数第K个节点

```javascript
var getKthFromEnd = function(head, k) {
    let p=head,q=head,i=0;
    while(p){
        if(i>=k){
            q=q.next;
        }
        p=p.next;
        i++;
    }
    return i<k?null:q;
};
```

#### 94.二叉树的中序遍历

```javascript
var inorderTraversal = function(root) {
    let res=[];
    let inOrder=function(node){
        if(!node){
            return null;
        }
        inOrder(node.left);
        res.push(node.val);
        inOrder(node.right);
    }
    inOrder(root);
    return res;
};
```

#### 21.合并两个有序链表

```javascript
var mergeTwoLists = function(l1, l2) {
    if(l1==null){
        return l2;
    }
    else if(l2==null){
        return l1;
    }
    else if(l1.val<l2.val){
        l1.next=mergeTwoLists(l1.next,l2);
        return l1;
    }
    else{
        l2.next=mergeTwoLists(l1,l2.next);
        return l2;
    }
};
```

#### 121买卖股票最佳时机

```javascript
var maxProfit = function(prices) {
    let min=prices[0],max=0;
    for(let i=0;i<prices.length;i++){
        if(prices[i]<min){
            min=prices[i];
            continue;
        }
        max=Math.max(prices[i]-min,max)
    }
    return max;
};
```

#### 160.相交链表

```javascript
var getIntersectionNode = function (headA, headB) {
    if(!headA||!headB) return null;
    let PA=headA,PB=headB;
    while(PA!=PB){
        PA=PA==null?headA:PA.next;
        PB=PB==null?headB:PB.next;
    }
    return PA;

}
```

#### 349.两个数组的交集

```javascript
var intersection = function (nums1, nums2) {
    nums1.sort((a, b) => a - b);
    nums2.sort((a, b) => a - b);
    let res=new Set();
    let i=0,j=0;
    while(i<nums1.length&&j<nums2.length){
        if(nums1[i]<nums2[j]){
            i++;
        } else if (nums1[i] > nums2[j]){
            j++;
        }else{
            res.add(nums1[i]);
            i++;
            j++;
        }
    }
    return [...res];
};
```



#### 54.螺旋矩阵

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



#### 209.长度最小的子数组

```javascript
var minSubArrayLen = function (target, nums) {
    let i=0,j=0,len=Infinity,sum=0;
    while(j<nums.length){
        sum+=nums[j];
        while(sum>=target){
            len=Math.min(len,j-i+1);
            sum-=nums[i];
            i++;
        }
        j++;
    }
    return len==Infinity?0:len;
};
```

