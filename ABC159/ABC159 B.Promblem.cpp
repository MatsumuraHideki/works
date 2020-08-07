#include <bits/stdc++.h>
using namespace std;
 
int main() {
  string s;
  cin >> s;
  
  
  int n=s.size();
  int sum1=0;
  int sum2=0;
  for(int i=0; i<(n-1)/2; i++){
  if(s.at(i)==s.at((n-3)/2-i) && s.at((n+1)/2+i)==s.at(n-1-i)){
    sum1++;
}
  }
    for(int j=0; j<n; j++){
    if(s.at(j)==s.at(n-1-j)){
      sum2++;
    }
    }
    
  if(sum1==(n-1)/2 && sum2==n){
    cout << "Yes" << endl;
  }
  else{
  cout << "No" << endl;
  }
}
