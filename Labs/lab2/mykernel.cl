__kernel void calculate_pi(
   int num_terms,
   __local float *local_result,
   __global float *global_result){
   
   /* work item index */
   int work_item_index = get_group_id(0);   

   /* initialize local data */
   local_result[work_item_index] = 0;

   /* Make sure previous processing has completed */
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform work item calculation */
   for(int i = 1; i < num_terms; i+=4){
      
      float term_1 = 1/(work_item_index*8+i);
      float term_2 = 1/(work_item_index*8+i+2);
      
      local_result[work_item_index] += (term_1 - term_2);
   } 

   /* Make sure previous processing has completed */
   barrier(CLK_LOCAL_MEM_FENCE);
   
   printf("Work item %d value: %f \n", work_item_index, local_result[work_item_index]);
   global_result[work_item_index] = local_result[work_item_index];

   return;

}
