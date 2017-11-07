void kernel space_to_batch(global float* space_data_ptr,
                           private const int space_batch,
                           private const int space_channel,
                           private const int space_height,
                           private const int space_width,
                           private const int block_height,
                           private const int block_width,
                           private const int b2s,
                           global float* batch_data_ptr) {
  int batch_idx = get_global_id(0);
  int batch_channel_idx = get_global_id(1);
  int batch_pixel_idx = get_global_id(2);

  const int batch_height = space_height / block_height;
  const int batch_width = space_width / block_width;
  const int batch_pixel_height_idx = batch_pixel_idx / batch_width;
  const int batch_pixel_width_idx = batch_pixel_idx % batch_width;

  const int block_size = block_height * block_width;
  const int space_idx = batch_idx / block_size;
  const int remaining_batch_idx = batch_idx % block_size;
  const int space_pixel_height_idx = (remaining_batch_idx / block_width) +
                                     batch_pixel_height_idx * block_height;
  const int space_pixel_width_idx = (remaining_batch_idx % block_width) +
                                     batch_pixel_width_idx * block_width;
  const int batch_data_offset = batch_idx * (space_channel * batch_height * batch_width) +
                                (batch_channel_idx * batch_height * batch_width) +
                                batch_pixel_height_idx * batch_width +
                                batch_pixel_width_idx;
  const int space_data_offset = space_idx * (space_channel * space_height * space_width) +
                                (batch_channel_idx * space_height * space_width) +
                                space_pixel_height_idx * space_width +
                                space_pixel_width_idx;
  if (b2s) {
    *(space_data_ptr + space_data_offset) = *(batch_data_ptr + batch_data_offset);
  } else {
    *(batch_data_ptr + batch_data_offset) = *(space_data_ptr + space_data_offset);
  }
}
