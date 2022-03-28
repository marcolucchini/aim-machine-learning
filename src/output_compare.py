# DO NOT MODIFY

import filecmp
  
def compare_outputs(f1,f2):
    result = filecmp.cmp(f1, f2, shallow=False)
    return result

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Input parameters.')
    parser.add_argument('path1', type=str, help='path to first .txt.')
    parser.add_argument('path2', type=str, help='path to first .txt.')
    args = parser.parse_args()
    globals().update(vars(args)) 
    print(compare_outputs(path1,path2))