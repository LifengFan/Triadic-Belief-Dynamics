void clustering()
{
   while(1){
   FindCandidateMergeInit();
   if (candidateMerge.size()<1) break;
   cout<<candidateMerge.size()<<endl;

   printf("Merge: %d %d\n", candidateMerge[0].first, candidateMerge[0].second);
   mergeAG(candidateMerge[0].first, candidateMerge[0].second);
   }



}