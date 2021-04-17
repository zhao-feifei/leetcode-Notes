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



