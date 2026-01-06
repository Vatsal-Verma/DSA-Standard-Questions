# infosys prep

---

# Arrays and String

### Combination Sum 
```

class Solution {
    public void track(int[] arr, int target, int index, List<Integer> lst, List<List<Integer>> result) {
        if(target == 0) {
            result.add(new ArrayList<>(lst));
            return;
        }

        if(target < 0) return ;
        for(int i=index; i<arr.length; i++ ) {  
            lst.add(arr[i]);

            track(arr, target - arr[i], i, lst, result);
            lst.remove(lst.size() - 1);
        }
    }

   
    public List<List<Integer>> combinationSum(int[] arr, int target) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> lst = new ArrayList<>();
        track(arr, target, 0, lst, result);
        return result;
    }
}

```
### Plus One
```
class Main {
    public static void main(String[] args) {
        int[] arr = {9, 9, 9};
        for(int i=arr.length - 1; i>=0; i -- ) {
            if(arr[i] < 9) {
                arr[i] ++;
                for(int j=0; j<arr.length; j++ ) System.out.print(arr[j] + " ");
                return;
            } else {
                arr[i] = 0;
            }
        }
        int[] brr = new int[arr.length + 1];
        brr[0] = 1;
        for(int i=0; i<brr.length; i++ ) System.out.print(brr[i] + " ");
    }
}
```

### Roman to Integer
```
class Solution {
    public int romanToInt(String s) {

        int result = 0;

        HashMap<Character, Integer> mp = new HashMap<>();
        mp.put('I', 1);
        mp.put('V', 5);
        mp.put('X', 10);
        mp.put('L', 50);
        mp.put('C', 100);
        mp.put('D', 500);
        mp.put('M', 1000);

        for(int i=0; i<s.length(); i++ ) {
            int curr = mp.get(s.charAt(i));
            int next = ((i + 1) < s.length()) ? mp.get(s.charAt(i + 1)) : 0;

            if(curr < next) {
                result -= curr;
            }

            else {
                result += curr;
            }
        }
        return result;

    }
}
```

### Integer to Roman

```
class Solution {
    public String intToRoman(int num) {
        int nums[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String som[] = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV","I"};
        
        StringBuilder sb = new StringBuilder();
        for(int i=0; i<nums.length; i++ ) {
            while(nums[i] <= num) {
                sb.append(som[i]);
                num -= nums[i];
            }
        }
        return sb.toString();
    }
}
```

### 2 Sum

```
class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> mp = new HashMap<>();
        for(int i=0; i<nums.length; i++ ) {
            int second = (target - nums[i]);
            if(mp.containsKey(second)) {
                return new int[] {i, mp.get(second)};
            }
            mp.put(nums[i], i);
        }
        return new int[]{0, 0};
    }
}
```

### 3 Sum

```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Set<List<Integer>> set = new HashSet<>();

        for(int i=0; i<nums.length; i++ ) {
            Set<Integer> lst = new HashSet<>();
            for(int j=i+1; j<nums.length; j ++) {
                int third = -(nums[i] + nums[j]);
                if(lst.contains(third)) {
                    List<Integer> temp = Arrays.asList(nums[i], nums[j], third);
                    Collections.sort(temp);
                    set.add(temp);
                } 
                lst.add(nums[j]);
            }
        }

        List<List<Integer>> result = new ArrayList<>(set);
        return result;

    }
}
```

### Group Anagrams
```
class Solution {
    public List<List<String>> groupAnagrams(String[] str) {
        HashMap<String, List<String>> mp = new HashMap<>();

        for(int i=0; i<str.length; i ++ ) {
            char[] ch = str[i].toCharArray();
            Arrays.sort(ch);
            String temp = String.valueOf(ch);

            mp.putIfAbsent(temp, new ArrayList<>());
            mp.get(temp).add(str[i]);
        }
        
        return new ArrayList<>(mp.values());
    }
}
```

### TOH

```
class Main {
    public static void toh(int n, char source, char helper, char destination) {   
        if(n == 0) return;
        
        toh(n - 1, source, destination, helper);
        System.out.println("source: " + source + ", destination: " + destination);
        
        toh(n - 1, helper, source, destination);
    }
    public static void main(String[] args) {
        int n = 3;
        toh(n, 'A', 'B', 'C');
    }
}
```
### Generate all permutations

```
class Solution {
    public void generate(int[] nums, int index, List<List<Integer>> result) {
        if(index == nums.length) {
            List<Integer> lst = new ArrayList<>();
            for(int i: nums) {
                lst.add(i);
            }
            result.add(lst);
            return;
        }

        for(int i=index; i<nums.length; i++ ) {
            swap(nums, index, i);
            generate(nums, index + 1, result);
            swap(nums, index, i);
        }
    }
    
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public List<List<Integer>> permute(int[] nums) {
        List<Integer> lst = new ArrayList<>();
        List<List<Integer>> result = new ArrayList<>();
        generate(nums, 0, result);
        return result;
    }
}
```
### Generate all the permuatations with n1, n2, n3 ... numbers of each character

```
class Main {
    public static List<String> result = new ArrayList<>();
    public static void generate(int a, int b, int c, String temp) {
        if(a == 0 && b == 0 && c == 0) {
            result.add(temp);
            return;
        }
        
        if(a > 0) generate(a - 1, b, c, temp + "a");
        if(b > 0) generate(a, b - 1, c, temp + "b");
        if(c > 0) generate(a, b, c - 1, temp + "c");
    }
    public static void main(String[] args) {
        String str = "abc";
        generate( 1, 1, 1, "");
        for(String i: result) {
            System.out.println(i);
        }
    }
}
```
### Find first and last occurence of an element
```
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int first = -1;
        int last = -1;

        for(int i=0; i<nums.length; i++ ) {
            if(nums[i] == target) {
                if(first == -1) {
                    first = i;
                }
                last = i;
            }
        }   
        
        return new int[] {first, last};
    }
}
```
### Spiral Matrix

```
class Solution {
    public List<Integer> spiralOrder(int[][] mat) {
        
        int row = mat.length;
        int col = mat[0].length;

        int left = 0;
        int right = col - 1;
        int top = 0;
        int bottom = row - 1;

        List<Integer> result = new ArrayList<>();

        while(left <= right && top <= bottom) {
            for(int i = left; i<=right; i++ ) {
                result.add(mat[top][i]);
            }
            top ++;

            for(int i=top; i<=bottom; i++ ) {
                result.add(mat[i][right]);
            }
            right --;

            if(top <= bottom) {
                for(int i=right; i>=left; i --) {
                    result.add(mat[bottom][i]);
                }
                bottom --;
            }

            if(left <= right) {
                for(int i=bottom; i>=top; i --) {
                    result.add(mat[i][left]);
                } 
            left ++;
            }
        }

        return result;
    }
}
```
### Rotate Array 

```
class Solution {
    public void rotate(int[] nums, int k) {
        k = k % nums.length;
        reverse(nums , 0 , nums.length-1);
        reverse(nums , 0 , k-1);
        reverse(nums , k , nums.length-1);

    }
    public void reverse(int[] nums, int start, int end){
        while(start < end){
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            end--;
            start++;
        }

    }
}
```
### Move all zero to end
```
class Solution {
    public void moveZeroes(int[] nums) {
        int a = 0;
        for(int i=0; i<nums.length; i++ ) {
            if(nums[i] != 0) nums[a ++] = nums[i];
        }

        for(int i=a; i<nums.length; i++ ) nums[i] = 0;
        
    }
}
```
### Maximum Product Subarray
```
Here we are supposed to find prefix product and suffix product, then we need to find maxium of them, if either prefix or suffix product becomes 0 then we update them to 1
```
```
class Solution {
    public int maxProduct(int[] nums) {
        int pre = 1;
        int suff = 1; 
        int maxi = 0;

        for(int i=0; i<nums.length; i++ ) {
            if(suff == 0) suff = 1;
            if(pre == 0) pre = 1;
            pre *= nums[i];
            suff *= nums[ nums.length - 1 - i ];
            maxi = Math.max(maxi, Math.max(pre, suff));
        }

        if(nums.length == 1) return nums[0];
        return maxi;
    }
}
```
### Kadane's algorithm (max subarray sum)

```
class Solution {
    public int maxSubArray(int[] nums) {
        long maxi = Integer.MIN_VALUE;
        long sum = 0;

        for(int i: nums) {
            sum += i;
            maxi = Math.max(maxi, sum);
            if(sum < 0) sum = 0;
        }
        return (int)maxi;
    }
}
```
### Next Greater Element 
```
class Solution {
    public static int[] nextGreaterElement(int[] arr) {
        int n = arr.length;
        int[] nge = new int[n];
        Stack<Integer> st = new Stack<>();
        for (int i = n - 1; i >= 0; i--) {

            while (!st.isEmpty() && st.peek() <= arr[i]) {
                st.pop();
            }
            nge[i] = st.isEmpty() ? -1 : st.peek();
            st.push(arr[i]);
        }
        return nge;
    }
```

### Best day to buy and sell stocks 

```
class Solution {
    public int maxProfit(int[] prices) {
        int mini = Integer.MAX_VALUE;
        int result = 0;

        for(int i=0; i<prices.length; i ++) {
            mini = Math.min(mini, prices[i]);
            result = Math.max(result, prices[i] - mini);
        }   

        return result;
    }
}
```

### Buy and Sell Stock II
```
We need to keep in mind that we chekc for concecutive elements and check i-1 th element is smaller than i, then we do sum += prices[i] - prices[i - 1]
```
```
class Solution {
    public int maxProfit(int[] prices) {
        int result = 0;
        for(int i=1; i<prices.length; i ++) {
            if(prices[i - 1] < prices[i]) {
                result += prices[i] - prices[i - 1];
            }
        }
        return result;
    }
}
```

### Print Pascal's Triangle 

```
class Main {
    public static void main(String[] args) {
        int n = 6;
        for(int i=1; i<=n; i ++) {
            int ans = 1;
            System.out.print("1 ");
            for(int j=1; j<i; j ++) {
                ans = ans * (i - j);
                ans = ans / j;
                System.out.print(ans +" ");
            }
            System.out.println();
        }
        
    }
}
```
```
output:
      1 
     1 1 
    1 2 1 
   1 3 3 1 
  1 4 6 4 1 
1 5 10 10 5 1 
```
### Print nth row of pascal's triangle
```
class Main {
    public static void main(String[] args) {
        int n = 6;  
            int ans = 1;
            System.out.print("1 ");
            for(int j=1; j<n; j ++) {
                ans = ans * (n - j);
                ans = ans / j;
                System.out.print(ans +" ");
         }
    }
}
```
```
output:
1 5 10 10 5 1 
```
### First Missing and repeating element
```
// Online Java Compiler
// Use this editor to write, compile and run your Java code online

class Main {
    public static void main(String[] args) {
        
        int arr[] = {1, 2, 2, 3, 4, 5, 7};
        
        int n = arr.length;
        
        int sn = (n * (n + 1) / 2);
        int ssq = (n * (n + 1) * (2 * n + 1) ) / 6;
        
        int s1 = 0;
        int s2 = 0;
        
        for(int i: arr) {
            s1 = s1 + i;
            s2 = s2 + (i * i);
        }
        
        int val1 = s1 - sn;
        int val2 = s2 - ssq;
        
        val2 = val2 / val1;
        
        int x = (val1 + val2) / 2;
        int y = x - val1;
        
        System.out.println(x + " " + y);
        
        
    }
}

```
```
output: 2 6
```

### Sort Elements by Frequency 
```
class Solution {
    public String frequencySort(String s) {
        HashMap<Character, Integer> mp = new HashMap<>();

        for (char ch : s.toCharArray()) {
            mp.put(ch, mp.getOrDefault(ch, 0) + 1);
        }

        PriorityQueue<Map.Entry<Character, Integer>> pq =
                new PriorityQueue<>((a, b) -> b.getValue() - a.getValue());

        pq.addAll(mp.entrySet());

        StringBuilder temp = new StringBuilder();
        while (!pq.isEmpty()) {
            Map.Entry<Character, Integer> entry = pq.poll();
            char ch = entry.getKey();
            int freq = entry.getValue();

            for (int i = 0; i < freq; i++) {
                temp.append(ch);
            }
        }

        return temp.toString();
    }
}

```
### Longest Pallindromic Substring (O(n^2))
```
class Solution {
    public boolean check(String s, int left, int right) {
       while(left < right) {
        if(s.charAt(left) != s.charAt(right)) {
            return false;
        }
        left ++;
        right --;
       }
       return true;
    }

    public String longestPalindrome(String s) {
        int start = 0;
        int maxLen = 0;

        for(int i = 0; i < s.length(); i++) {
            for(int j = i; j < s.length(); j++) {
                if((j - i + 1) > maxLen && check(s, i, j)) {
                    maxLen = j - i + 1;
                    start = i;
                }  
            }
        }
        return s.substring(start, start + maxLen);
    }   
}

```
# LinkedList

### Middle of a LinkedList
```
class Solution {
    public ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while(fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next; 
        }
        return slow;
    }
}
```
### Reverse a LinkedList
```
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode temp = head;
        ListNode prev = null;

        while(temp != null) {
            ListNode nextNode = temp.next;
            temp.next = prev;
            prev = temp;
            temp = nextNode;
        }

        return prev;
    }
}
```

### Detect Loop in a linkedList 
```
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;

        while(fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if(slow == fast) {
                return true;
            }
        }
        return false;
    }
}
```
### seperate Odd and even indexes
```
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode odd = head;
        ListNode even = head.next;
        ListNode evenHead = head.next;

        while(even != null && even.next != null) {
            odd.next = odd.next.next;
            even.next = even.next.next;

            odd = odd.next;
            even = even.next;
        }

        odd.next = evenHead;
        return head;
    }
}
```
### find intersection of two linkedList
```
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode temp1 = headA;
        ListNode temp2 = headB;

        while(temp1 != temp2) {
            temp1 = temp1.next;
            temp2 = temp2.next;

            if(temp1 == temp2) return temp1;

            if(temp1 == null) temp1 = headB;
            if(temp2 == null) temp2 = headA;
        }
        return temp1;
    }
}
```
### Add Two Numbers in a linnkedList
```
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode temp = dummy;

        int carry = 0;
        while(l1 != null || l2 != null) {
            int sum = 0;
            if(l1 != null) {
                sum += l1.val;
                l1 = l1.next;
            }

            if(l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }

            sum += carry;
            temp.next = new ListNode(sum % 10);
            carry = sum / 10;
            temp = temp.next;
        }

        if(carry != 0) {
            temp.next = new ListNode(carry);
        }

        return dummy.next;
    }
}
```
### Merge two sorted LinkedList
```
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode(0);
        ListNode temp = dummy;
        while(list1 != null && list2 != null) {
            if(list1.val < list2.val) {
                temp.next = new ListNode(list1.val);
                temp = temp.next;
                list1 = list1.next;
            }
            else {
                temp.next = new ListNode(list2.val);
                temp = temp.next;
                list2 = list2.next;
            }
        }

        if(list1 != null) temp.next = list1;
        if(list2 != null) temp.next = list2;

        return dummy.next;
    }
}
```
# Sliding Window (Varible Size)

### Longest Subarray without repeating characters
```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int maxLen = 0;
        int right = 0;
        int left = 0;
        HashMap<Character, Integer> mp = new HashMap<>();

        while(right < s.length()) {
            mp.put(s.charAt(right), mp.getOrDefault(s.charAt(right), 0) + 1);

            while(mp.get(s.charAt(right)) > 1) {
                mp.put(s.charAt(left), mp.get(s.charAt(left)) - 1);
                left ++;
            }

            maxLen = Math.max(maxLen, right - left + 1);
            right ++;
        }
        return maxLen;
    }
}
```
### maximum concecutive ones iii
```
class Solution {
    public int longestOnes(int[] nums, int k) {
        int zero = 0;
        int result = 0;
        int left = 0;
        int right = 0;
        

        while(right < nums.length) {
            if(nums[right] == 0) {
                zero ++;
            }
            
            while(zero > k) {
                if(nums[left] == 0) {
                    zero --;
                }
                left ++;
            }

            result = Math.max(result, right - left + 1);
            right ++;

        }
        return result;
    }
}
```

### Binary Subarray with sum
```
class Solution {
    public int subArray(int[] nums, int goal) {
        int right = 0;
        int left = 0;
        int result = 0;
        int sum = 0;

        while(right < nums.length) {
            sum += nums[right];
            
            while(sum > goal) {
                sum -= nums[left];
                left ++;
            }

            result += right - left + 1;
            right ++;
        }
        return result;
    }
    public int numSubarraysWithSum(int[] nums, int goal) {
        return subArray(nums, goal) - subArray(nums, goal - 1);
    }
}
```

### Count number of nice subarrays
```
class Solution {
    public int check(int[] nums, int k) {
        int left = 0;
        int right = 0;
        int sum = 0;
        int count = 0;

        while(right < nums.length) {
            sum += nums[right] % 2;
            while(sum > k) {
                sum -= nums[left] % 2;
                left ++;
            }
            count += right - left + 1;
            right ++;
        }   

        return count;
    }
    public int numberOfSubarrays(int[] nums, int k) {
        return check(nums, k) - check(nums, k - 1);
    }
    
}
```
### Count substring containing all the three characters
```
class Solution {
    public boolean check(String str) {
        return str.contains("a") && str.contains("b") && str.contains("c");
    }
    public int numberOfSubstrings(String s) {
        int right = 0;
        int left = 0;
        int count = 0;

        while(right < s.length()) {
            right ++;
            while(check(s.substring(left, right))) {
                count += s.length() - right + 1;
                left ++;
            }
        }
        return count;
    }
}
```

### Longest substring length with K distinct characters
```
class Solution {
    public int longestKSubstr(String s, int k) {
        int left = 0, right = 0;
        int maxLen = -1;

        HashMap<Character, Integer> mp = new HashMap<>();

        while (right < s.length()) {
            char ch = s.charAt(right);
            mp.put(ch, mp.getOrDefault(ch, 0) + 1);

            while (mp.size() > k) {
                char leftChar = s.charAt(left);
                mp.put(leftChar, mp.get(leftChar) - 1);
                if (mp.get(leftChar) == 0) {
                    mp.remove(leftChar);
                }
                left++;
            }
            if (mp.size() == k) {
                maxLen = Math.max(maxLen, right - left + 1);
            }

            right++;
        }

        return maxLen;
    }
}

```

### Subarrays with K Different Integers
```
class Solution {
    public int countSub(int[] nums, int k) {
        int left = 0;
        int right = 0;
        int count = 0;

        HashMap<Integer, Integer> mp = new HashMap<>();
        while(right < nums.length) {
            mp.put(nums[right], mp.getOrDefault(nums[right], 0) + 1);
            while(mp.size() > k) {
                mp.put(nums[left], mp.get(nums[left]) - 1);
                if(mp.get(nums[left]) == 0) mp.remove(nums[left]);
                left ++;
            }
            count += right - left + 1;
            right ++;
        }
        return count;
    }
    public int subarraysWithKDistinct(int[] nums, int k) {
        return countSub(nums, k) - countSub(nums, k - 1);
    }
}
```
# Greedy Based Questions
### Assign Cookies
```
class Solution {
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);

        int left = 0;
        int right = 0;

        while(left < s.length && right < g.length) {
            if(g[right] <= s[left]) {
                right ++;
            }
            left ++;
        }

        return right;
    }
}
```

### Minimum Number of coins 
```
class Main {
    public static void main(String[] args) {
        int[] nums = {1, 2, 5, 10, 20, 50, 100, 500, 1000};
        int target = 555;
        
        List<Integer> result = new ArrayList<>();
        
        for(int i=nums.length - 1; i>=0; i --) {
            while(target >= nums[i]) {
                target -= nums[i];
                result.add(nums[i]);
            }
        }
        
        result.forEach(n -> System.out.print(n + " "));
    }
}
```
### fractional Knapsack

```
import java.util.*;
class Main {
    public static void main(String[] args) {
        double result = 0;
        int arr[][] = {{60, 10}, {100, 20}, {120, 30}};
        int cap = 50;
        
        Arrays.sort(arr, (a, b) ->
            Double.compare((double) b[0] / b[1], (double) a[0] / a[1])
        );
        
        for(int i = 0; i < arr.length; i++) {
            if(cap >= arr[i][1]) {
                result += arr[i][0];
                cap -= arr[i][1];
            } else {
                result += ((double) arr[i][0] / arr[i][1]) * cap;
                break; 
            }
        }
        
        System.out.println(result);
    }
}
```
### Lemonade Change
```
class Solution {
    public boolean lemonadeChange(int[] bills) {

        int five = 0;
        int ten = 0;
        int twen = 0;

        for(int i=0; i<bills.length; i ++ ){ 
            if(bills[i] == 5) {
                five ++;
            }

            if(bills[i] == 10) {
                if(five > 0) {
                    five --;
                    ten ++;
                }
                else {
                    return false;
                }
            }

            if(bills[i] == 20) {
                if(five > 0 && ten > 0) {
                    five --;
                    ten --;
                }
                else if(five >= 3) {
                    five -= 3;
                }
                else 
                return false;
            }
        }
        return true;
    }
}
```
### Valid Parenthesis Checker String
```
class Solution {
    public boolean checkValidString(String s) {

        int min = 0;
        int max = 0;
        
        for(int i=0; i<s.length(); i ++) {
            if(s.charAt(i) == '(') {
                max ++;
                min ++;
            }
            else if(s.charAt(i) == ')') {
                max --;
                min --;
            }
            else {
                min --;
                max ++;
            }
            if(min < 0) min = 0;
            if(max < 0) return false;
        }
        return min == 0;
    }
}
```

### Jump game I
```
class Solution {
    public boolean canJump(int[] nums) {
        int far = 0;
        for(int i=0; i<nums.length; i ++ ) {
            if(i > far) return false;
            far = Math.max(far, nums[i] + i);
        }

        return true;
    }
}
```
### Jump Game II
```
class Solution {
    public int jump(int[] nums) {

        int left = 0;
        int right = 0;
        int far = 0;
        int jumps = 0;

        while(right < nums.length - 1) {
            far = 0;
            for(int i=left; i<=right; i++ ) {
                far = Math.max(far, nums[i] + i);
            }

            left = right + 1;
            right = far;
            jumps ++;
        }

        return jumps;
    }
}
```
### Candy 
```
class Solution {
    public int candy(int[] arr) {
        int n = arr.length;
        int sum = 1;
        int i = 1;

        while(i < n) {
            while(i < n && arr[i] == arr[i - 1]) {
                i ++;
                sum ++;
                continue;
            }

            int peak = 1;
            while(i < n && arr[i] > arr[i - 1]) {
                peak ++;
                sum += peak;
                i ++;
            }

            int down = 1;
            while(i < n && arr[i] < arr[i - 1]) {
                sum += down;
                down ++;
                i ++;
            }

            if(down > peak) {
                sum += (down - peak);
            }
        }
        return sum;
    }
}
```
### Merge Intervals
```
class Solution {
    public int[][] merge(int[][] intervals) {
        
       List<int[]> result = new ArrayList<>();
       Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
       int start = intervals[0][0];
       int end = intervals[0][1];

       for(int[] i: intervals) {
            if(i[0] <= end) {
                end = Math.max(end, i[1]);
            }
            else {
                result.add(new int[]{start, end});
                start = i[0];
                end = i[1];
            }
       }
       result.add(new int[]{start, end});
       return result.toArray(new int[0][]);
    }
}
```
### Non Overlapping intervals
```
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        int count = 1;
        Arrays.sort(intervals, (a, b) -> a[1] - b[1]);

        int start = intervals[0][0];
        int end = intervals[0][1];

        for(int i=0; i<intervals.length; i ++ ) {
            if(intervals[i][0] >= end) {
                end = intervals[i][1];
                count ++;
            }
        }
        return intervals.length - count;
    }
}
```
### Insert Intervals
```
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
   
        List<int[]> result = new ArrayList<>();
        int i = 0;
        int n = intervals.length;

        while(i < n && intervals[i][1] < newInterval[0]) {
            result.add(intervals[i]);
            i ++;
        }

        while(i < n && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
            newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
            i ++;
        }
        result.add(newInterval);

        while(i < n) {
            result.add(intervals[i]);
            i ++;
        }

        return result.toArray(new int[0][]);
    }
}
```

# Binary Tree 

### Diameter of the binary Tree
```
// left subtree depth + right subtree depth
class Solution {
    public int diameter = 0;
    public int height(TreeNode root) {
          if(root == null) return 0;

        int lh = height(root.left);
        int rh = height(root.right);

        diameter = Math.max(diameter, lh + rh);

        return 1 + Math.max(lh, rh);
    }
    public int diameterOfBinaryTree(TreeNode root) {
        int k = height(root);
        return diameter;
    }
}
```
### Binary Tree Maximum Path Sum
```
class Solution {
    int maxPath = Integer.MIN_VALUE;

    public int dfs(TreeNode root) {
        if (root == null) return 0;

        int lh = Math.max(0, dfs(root.left));
        int rh = Math.max(0, dfs(root.right));

        maxPath = Math.max(maxPath, root.val + lh + rh);
        return root.val + Math.max(lh, rh);
    }

    public int maxPathSum(TreeNode root) {
        dfs(root);
        return maxPath;
    }
}

```

### Zig-Zag Traversal
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        boolean check = true;

        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        Queue<TreeNode> nq = new LinkedList<>();
        nq.add(root);

        while(!nq.isEmpty()) {
            int size = nq.size();
            List<Integer> row = new ArrayList<>();
            for(int i=0; i<size; i++ ) {
                TreeNode node = nq.poll();
                if(!check) {
                    row.add(0, node.val);
                }
                else {
                    row.add(node.val);
                }

                if(node.left != null) nq.add(node.left);
                if(node.right != null) nq.add(node.right);
            }

            check = !check;
            result.add(row);
        }

        return result;
    }
}
```

### top view of Binary Tree
```
class Solution {
    class Pair {
        Node node; 
        int pos; 
        Pair(Node node, int pos) {
            this.node = node;
            this.pos = pos;
        }
    }
    public ArrayList<Integer> topView(Node root) {
        ArrayList<Integer> result = new ArrayList<>();
        if(root == null) return result;
        
        Queue<Pair> nq = new LinkedList<>();
        nq.add(new Pair(root, 0));
        
        TreeMap<Integer, Integer> mp = new TreeMap<>();
        
        while(!nq.isEmpty()){
            
            Pair pair = nq.poll();
            Node ver = pair.node;
            int pos = pair.pos;
            
            if(!mp.containsKey(pos)) {
                mp.put(pos, ver.data);
            } 
            
            if(ver.left != null) {
                nq.add(new Pair(ver.left, pos - 1));
            }
            
            if(ver.right != null) {
                nq.add(new Pair(ver.right, pos + 1));
            }
        }
        
        for(int i: mp.keySet()) {
            result.add(mp.get(i));
        }
        
        return result;
    }
}
```

### Path Sum I
```
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if(root == null) return false;
        if(root.left == null && root.right == null) return targetSum == root.val;
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }
}
```
### Path Sum II
```
class Solution {
    public void dfs(TreeNode node, int targetSum, List<Integer> path, List<List<Integer>> result, int sum) {
        if(node == null) return;
        sum += node.val;
        path.add(node.val);
        if(node.left == null && node.right == null && sum == targetSum) {
            result.add(new ArrayList<>(path));
        } else {
            dfs(node.left, targetSum, path, result, sum);
            dfs(node.right, targetSum, path, result, sum);
        }
        path.remove(path.size() - 1);
    }
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs(root, targetSum, path, result, 0);
        return result;
    }
}
```
### Lowest Common Ancestor
```

class Solution {
    public boolean dfs(TreeNode root, TreeNode target, List<TreeNode> lst) {
        if(root == null) return false;
        lst.add(root);

        if(root == target) return true;

        if(dfs(root.left, target, lst) || dfs(root.right, target, lst)) return true;

        lst.remove(lst.size() - 1);
        return false;
    }
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        List<TreeNode> lst1 = new ArrayList<>();
        List<TreeNode> lst2 = new ArrayList<>();

        dfs(root, p, lst1);
        dfs(root, q, lst2);

        int i = 0;
        
        while(i < lst1.size() && i < lst2.size() && lst1.get(i) == lst2.get(i)) {
            i ++;
        }

        return lst1.get(i - 1);

    }
}
```
### Maximum width of the binary tree
```
class Pair{
    TreeNode node;
    long index;
    Pair(TreeNode node, long index) {
        this.node = node;
        this.index = index;
    }
}
class Solution {
    public int widthOfBinaryTree(TreeNode root) {
        if(root == null) return 0;
        long result = 0;

        Queue<Pair> nq = new LinkedList<>();
        nq.add(new Pair(root, 0));

        while(!nq.isEmpty()) {
            long size = nq.size();
            long min = nq.peek().index; 
            long first = 0;
            long last = 0;

            for(int i=0; i<size; i ++) {
                Pair p = nq.poll();
                long index = p.index - min;

                if(i == 0) first = index;
                if(i == size - 1) last = index;

                if(p.node.left != null) {
                    nq.add(new Pair(p.node.left, 2 * index + 1));
                }
                
                if(p.node.right != null) {
                    nq.add(new Pair(p.node.right, 2 * index + 2));
                }
            }

            result = Math.max(result, last - first + 1);
        }

        return (int)result;
    }
}
```

# Graph
### Number of provinces (Number of connected componenets)
```
class Solution {
    public void dfs(int node, List<List<Integer>> adj, int[] vis) {
        vis[node] = 1;
        for(int i: adj.get(node)) {
            if(vis[i] == 0)
            dfs(i, adj, vis);
        }
    }
    public int findCircleNum(int[][] mat) {
        int v = mat[0].length;
        List<List<Integer>> adj = new ArrayList<>();
        for(int i=0; i<v; i++ ) {
            adj.add(new ArrayList<>());
        }

        for(int i=0; i<v; i++ ) {
            for(int j=0; j<v; j ++) {
                if(mat[i][j] == 1) {
                adj.get(i).add(j);
                adj.get(j).add(i);
                }
            }
        }

        int[] vis = new int[v];

        int count = 0;
        for(int i=0; i<v; i++ ) {
            if(vis[i] == 0) {
                dfs(i, adj, vis);
                count ++;
            }
        }
        return count;
    }
}
```
### Rotten Oranges
```

class Pair {
    int row;
    int col;
    int time; 
    Pair(int row, int col, int time) {
        this.row = row;
        this.col = col;
        this.time = time;
    }
}
class Solution{
    public int orangesRotting(int[][] grid) {
        Queue<Pair> nq = new LinkedList<>();
        int n = grid.length;
        int m = grid[0].length;

        int[][] vis = new int[n][m];
        int fresh = 0;
        int afterfresh = 0;

        for(int i=0; i<n; i++ ) {
            for(int j=0; j<m; j ++) {
                if(grid[i][j] == 2) {
                    nq.add(new Pair(i, j, 0));
                    vis[i][j] = 2;
                }
                if(grid[i][j] == 1) {
                    fresh ++;
                }
            }
        }

        int res = 0;
        

        int[] dRow = {0, 1, 0, -1};
        int[] dCol = {-1, 0, 1, 0};

        while(!nq.isEmpty()) {
            Pair p = nq.poll();
            int row = p.row;
            int col = p.col;
            int time = p.time;

            res = Math.max(res, time);

            for(int i=0; i<4; i ++) {
                int nrow = row + dRow[i];
                int ncol = col + dCol[i];

                if(nrow < n && nrow >= 0 && ncol < m && ncol >= 0 && grid[nrow][ncol] == 1 && vis[nrow][ncol] == 0) {
                    nq.add(new Pair(nrow, ncol, time + 1));
                    vis[nrow][ncol] = 2;
                    afterfresh ++;
                }
            }
        }

        if(fresh != afterfresh) return -1;
        return res;
    }
}
```
### Flood Fill
```
class Solution {
    public void flood(int row, int col, int[][] image, int[][] vis, int color, int iniColor, int[] dRow, int[] dCol) {
        vis[row][col] = color;
        int n = image.length;
        int m = image[0].length;

        for(int i=0; i<4; i++) {
            int nrow = row + dRow[i];
            int ncol = col + dCol[i];

            if(nrow < n && nrow >=0 && ncol < m && ncol >= 0 && image[nrow][ncol] == iniColor && vis[nrow][ncol] != color) {
                flood(nrow, ncol, image, vis, color, iniColor, dRow, dCol);
            }
        }
    }
    public int[][] floodFill(int[][] image, int sr, int sc, int color) {
        int iniColor = image[sr][sc];
        int n = image.length;
        int m = image[0].length;
        int vis[][] = image;
        int[] dRow = {0, 1, 0, -1};
        int[] dCol = {-1, 0, 1, 0};

        flood(sr, sc, image, vis, color, iniColor, dRow, dCol);

        return vis;
    }
}
```
### Detect cycle in undirected graph (BFS)
```class Pair {
    int node;
    int parent;
    Pair(int node, int parent) {
        this.node = node;
        this.parent = parent;
    }
}
class Solution {
    public boolean dfs(int node, int parent, List<List<Integer>> adj, int[] vis) {
       Queue<Pair> nq = new LinkedList<>();
       nq.add(new Pair(node, -1));
       vis[node] = 1;
       while(!nq.isEmpty()) {
           Pair p = nq.poll();
           int ver = p.node;
           int par = p.parent;
           
           for(int i: adj.get(ver)) {
               if(vis[i] == 0) {
                   nq.add(new Pair(i, ver));
                   vis[i] = 1;
               }
               else if(par != i) return true;
           }
       }
           return false;
    }

    public boolean isCycle(int v, int[][] edges) {
        List<List<Integer>> adj = new ArrayList<>();

        for (int i = 0; i < v; i++) adj.add(new ArrayList<>());

        for (int[] i : edges) {
            int u = i[0];
            int v1 = i[1];

            adj.get(u).add(v1);
            adj.get(v1).add(u);
        }

        int[] vis = new int[v];

        for (int i = 0; i < v; i++) {
            if (vis[i] == 0) {
                if (dfs(i, -1, adj, vis)) return true;
            }
        }
        return false;
    }
}
```

### Detect cycle in undirected Graph (DFS)
```
class Solution {
    public boolean dfs(int node, int parent, List<List<Integer>> adj, int[] vis) {
        vis[node] = 1;

        for (int i : adj.get(node)) {
            if (vis[i] == 0) {
                if (dfs(i, node, adj, vis)) return true;
            } 
            else if (i != parent) { 
                return true; 
            }
        }
        return false;
    }

    public boolean isCycle(int v, int[][] edges) {
        List<List<Integer>> adj = new ArrayList<>();

        for (int i = 0; i < v; i++) adj.add(new ArrayList<>());

        for (int[] i : edges) {
            int u = i[0];
            int v1 = i[1];

            adj.get(u).add(v1);
            adj.get(v1).add(u);
        }

        int[] vis = new int[v];

        for (int i = 0; i < v; i++) {
            if (vis[i] == 0) {
                if (dfs(i, -1, adj, vis)) return true;
            }
        }
        return false;
    }
}

```

### Surrounded Regions
```
class Solution {
    public void dfs(int row, int col, char[][] mat, int[][] vis, int[] dRow, int[] dCol) {
        vis[row][col] = 1;  

        int n = mat.length;
        int m = mat[0].length;
        for(int i=0; i<4; i++) {
            int nrow = row + dRow[i];
            int ncol = col + dCol[i];

            if(nrow < n && nrow >=0 && ncol < m && ncol >=0 && mat[nrow][ncol] == 'O' && vis[nrow][ncol] == 0) {
                dfs(nrow, ncol, mat, vis, dRow, dCol);
            }
        }
    }
    public void solve(char[][] mat) {
        int n = mat.length;
        int m = mat[0].length;

        int vis[][] = new int[n][m];
        
        int[] dRow = {0, 1, 0, -1};
        int[] dCol = {-1, 0, 1, 0};

        for(int i=0; i<n; i ++ ) {
            if(mat[i][0] == 'O' && vis[i][0] == 0) {
                dfs(i, 0, mat, vis, dRow, dCol);
            }

            if(mat[i][m - 1] == 'O' && vis[i][m - 1] == 0) {
                dfs(i, m - 1, mat, vis, dRow, dCol);
            }
        }

        for(int i=0; i<m; i++ ) {
            if(mat[0][i] == 'O' && vis[0][i] == 0) {
                dfs(0, i, mat, vis, dRow, dCol);
            }

            if(mat[n - 1][i] == 'O' && vis[n - 1][i] == 0) {
                dfs(n - 1, i, mat, vis, dRow, dCol);
            }
        }

        for(int i=0; i<n; i++) {
            for(int j=0; j<m; j ++) {
                if(mat[i][j] == 'O' && vis[i][j] == 0) {
                    mat[i][j] = 'X';
                } 
            }
        }
        
    }
}
```
### Is Graph Bipartite?
```
class Solution {
    public boolean dfs(int node, int[] color, int col, int[][] graph) {
        color[node] = col;
        
        for(int index: graph[node]) {
            if(color[index] == -1) {
                if(!dfs(index, color, 1 - col, graph)) return false;
            }

            else if(color[index] == col) return false;
        }
        return true;
    }
    public boolean isBipartite(int[][] graph) {

        int v = graph.length;
        int[] color = new int[v];
        Arrays.fill(color, -1);

        for(int i=0; i<v; i++ ) {
            if(color[i] == -1)
            if(!dfs(i, color, 0, graph)) return false;
        }

        return true;
    }
}
```

### Detect Cycle in a Directed Acyclic Graph or Directed Graph (BFS approack i.e. topo sort, kahn's algorithm)
```
class Solution {
    public boolean isCyclic(int v, int[][] edges) {
        // code here
        List<List<Integer>> adj = new ArrayList<>();
        for(int i=0; i<v; i++ ) adj.add(new ArrayList<>());
        
        for(int[] i: edges) {
            int u = i[0];
            int v1 = i[1];
            
            adj.get(u).add(v1);
        }
        
        int[] indegree = new int[v];
        
        for(int i=0; i<v; i++ ) {
            for(int j: adj.get(i)) {
                indegree[j] ++;
            }
        }
        
        Queue<Integer> nq = new LinkedList<>();
        
        for(int i=0; i<v; i++ ) {
            if(indegree[i] == 0) {
                nq.add(i);
            }
        }
        
        int check = 0;
        
        while(!nq.isEmpty()) {
            int node = nq.poll();
            check ++;
            for(int i: adj.get(node)) {
                indegree[i] --;
                if(indegree[i] == 0) {
                    nq.add(i);
                }
            }
        }
        
        if(check != v) return true;
        return false;
    }
}
```
### Cycle Detection in Directed Graph (DFS using vis and pathVis)
```
class Solution {
    public boolean dfs(int node, List<List<Integer>> adj, int[] vis, int[] pathVis) {
        vis[node] = 1;
        pathVis[node] = 1;
        
        for(int i: adj.get(node)) {
            if(vis[i] == 0) {
                if(dfs(i, adj, vis, pathVis)) return true;
            }
            
            else if(pathVis[i] == 1) return true;
        }
        
        pathVis[node] = 0;
        return false;
    }
    public boolean isCyclic(int v, int[][] edges) {
        // code here
        int vis[] = new int[v];
        int pathVis[] = new int[v];
        
        List<List<Integer>> adj = new ArrayList<>();
        
        for(int i=0; i<v; i++ ) adj.add(new ArrayList<>());
        
        for(int[] i: edges) {
            int u = i[0];
            int v1 = i[1];
            adj.get(u).add(v1);
        }
        
        for(int i=0; i<v; i++ ) {
            if(vis[i] == 0) {
                if(dfs(i, adj, vis, pathVis)) return true;
            }
        }
        
        return false;
    }
}
```
