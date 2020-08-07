#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int n;
  cin >> n;
  
  vector<int> a(n);
  for (int i = 0; i < n; i++) {
    cin >> a[i];
}
  
  vector<int> num(n);
  for(int j=0; j<n-1; j++){
    num[a[j]-1]++;
  }
  
  for(int k=0; k<n; k++){
    cout << num[k] << endl;
  }
  
}
