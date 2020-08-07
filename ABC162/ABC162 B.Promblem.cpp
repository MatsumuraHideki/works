#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int n;
  cin >> n;
  
  int64_t num=0;
  int i=1;
  while(i<n+1){
    if(i%3!=0){
      if(i%5!=0){
      num = num + i;
      }
    }
    i++;
  }
  
 cout << num << endl;
  
  }
