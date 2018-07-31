class GotIt_config(object):
  valid_set_idx = [9, 10, 28, 31, 32, 40, 52, 53, 59, 69, 71, 77, 79, 89]
  train_set_idx = [41, 56, 27, 11, 64, 87, 54, 25, 82, 33, 99,  3,  6, 74, 36, 68, 51,
        2, 49, 92, 63, 78, 58, 95,  1, 66, 21, 22, 67, 72, 39, 94, 14, 20,
       17, 60, 97, 18, 50, 85, 43, 47,  8, 76, 19]
  test_set_idx = [93, 46, 29, 80, 23, 96,
       16,  5,  7, 48, 38, 73, 83, 44, 15, 30, 98, 81, 91, 55, 45]
  gotit_role_dict = {'Expert':1, 'Student':-1}
  # act_id_map = {1: 'Provide-Info', 2:'Request-Info', 3:'Request-Action', 4:'Greet/Ack'}
  act_id_map = {1: 'Provide-Info', 2:'Request-Info', 3:'Greet/Ack'}
