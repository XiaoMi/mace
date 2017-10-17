void kernel simple_add(global const int *a,
                       global const int *b,
                       global int *c,
                       global const int *step) {
  int id = get_global_id(0);
  int start = step[0] * id;
  int stop = start + step[0];
  for (int i = start; i < stop; i++) c[i] = a[i] + b[i];
}
