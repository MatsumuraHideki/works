#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int n;
  cin >> n;
  
  vector<int> x(n);
  for(int i=0; i<n; i++){
    cin >> x[i];
  }
 
  priority_queue<int> pq;
  for(int j=1; j<101; j++){
    int sum=0;
  for(int k=0; k<n; k++){
    sum = sum + ((j-x[k])*(j-x[k]));
  }
    pq.push(sum);
  }
  
  while(pq.size()!=1){
    pq.pop();
  }
  
  cout << pq.top() << endl;
    
}
