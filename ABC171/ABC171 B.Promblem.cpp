#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int n, k;
  cin >> n >> k;
  
  vector<int> a(n);
  for(int i=0; i<n; i++){
    cin >> a[i];
  }
  
  int sum=0;
  sort(a.begin(), a.end());
  for(int j=0; j<k; j++){
    sum += a[j];
  }
  
  cout << sum << endl;
  
  
}