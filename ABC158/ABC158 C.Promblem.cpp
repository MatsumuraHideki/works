#include <bits/stdc++.h>
using namespace std;
 
int main() {
int a, b;
  cin >> a >> b;
  
  priority_queue<int> pq;
  int i=0;
  
  while(pq.size()==0 && i<10001){
    i++;
      if(i*8/100 == a && i/10 == b){
        pq.push(i);
      }
    }
  
  if(pq.size()!=0){
  cout << pq.top() << endl;
  }
  else if(pq.size()==0) {
    cout << -1 << endl;
  }
  
}
