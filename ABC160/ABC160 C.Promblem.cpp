#include <bits/stdc++.h>
using namespace std;
 
int main() {
int k, n;
  cin >> k >> n;
  
  vector<int> A(n);
  for(int i=0; i<n; i++){
    cin >> A[i];
  }
  
  vector<int> d(n);
  d[0]=max(A[1]-A[0], A[2]-A[1]);
  
  for(int j=1; j<n-2; j++){
    d[j]=max(d[j-1], A[j+2]-A[j+1]);
  }
  
  d[n-2]=max(d[n-3], k-A[n-1]+A[0]);
  
  cout << k-d[n-2] << endl;
  
}
