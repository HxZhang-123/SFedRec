def parameter_clip(parameter_list):
    clipped_para = []
    none_positions = []
    grad_list = []
    for single_param in parameter_list:
        none_position = [i for i, val in enumerate(single_param[0]) if val is None]
        modified_list = [x for x in single_param[0] if x is not None]
        modified_tuple = (modified_list, single_param[1], single_param[2])
        clipped_para.append(modified_tuple)
        none_positions.append(none_position)
        grad_list.append(modified_list)
    return clipped_para, none_positions, grad_list

def gen_private_param(model_grad):
    private_grad = model_grad[24]
    return private_grad

def filter_clipped_param_by_length(clipped_param):
    filtered_param = []
    index = []
    for i in range(len(clipped_param)):
        if len(clipped_param[i][0]) == 37:
            filtered_param.append(clipped_param[i])
        else:
            index.append(i)
    return filtered_param, index

def del_wrong_client(uncommon, grad_list, none_positions):
    del_list = list(reversed(uncommon))
    new_none_positions_list = []
    new_grad_list = []
    new_grad_list = [sublist for i, sublist in enumerate(grad_list) if i not in del_list]
    new_none_positions_list = [sublist for i, sublist in enumerate(none_positions) if i not in del_list]
    return new_grad_list, new_none_positions_list