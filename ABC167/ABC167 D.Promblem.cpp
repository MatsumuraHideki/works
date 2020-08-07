#include <bits/stdc++.h>
using namespace std;
 
int main() {
 int64_t n, k;
  cin >> n >> k;
  
  vector<int> A(n);
  
  for (int p = 0; p < n; p++) {
    cin >> A.at(p);
  }
  
    vector<int> B(n);
   B.at(0)=1; 
  
   set<int> S;
   S.insert(1);
  
  int q=0;
  int i=1;
  while(q==0){
    B.at(i)=A.at(B.at(i-1)-1);
    if (S.count(B.at(i))) {
      q++;
    }
    S.insert(B.at(i));
    i++;
    }

  
    
  int j=0;
  int m=0;
  while(m==0){
    if(B.at(j)==B.at(i-1)){
      m++;
    }
    j++;
  }
  
  
  if(k>=i){
  cout << B.at(j+((k-j)%(i-j))) << endl;
  }
  
  else{
    cout << B.at(k) << endl;
  }


}
