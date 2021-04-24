## 容易

415.字符串相加

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



88.合并两个有序数组

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

1.两数之和

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

70.爬楼梯

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

206.反转链表