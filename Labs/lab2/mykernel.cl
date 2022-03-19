__kernel void calculate_pi(
   int num_terms,
   __local float *local_result,
   __global float *global_result){
   
   /* initialize local data */
   local_result[get_local_id(0)] = 0;

   /* Make sure previous processing has completed */
   barrier(CLK_LOCAL_MEM_FENCE);
   
   global_result[get_local_id(0)] = 1.0;

}
