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
