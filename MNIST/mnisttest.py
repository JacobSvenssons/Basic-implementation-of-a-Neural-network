import sys

def run():
    
    if len(sys.argv) != 3:
        print("Usage: python mnisttest <result label file> <validation label file>")
        exit()
    
    resultlabelfile = open(sys.argv[1])    
    validationlabelfile = open(sys.argv[2])
    
    validationlabelfile.readline()
    validationlabelfile.readline()
    [nlabels, digits] = validationlabelfile.readline().split()
    nlabels = int(nlabels) 
    
    h = 0
    
    for i in range(0, nlabels):
        lr = resultlabelfile.readline()
        lv = validationlabelfile.readline()
        if int(lr) == int(lv):
            h = h + 1

    p = ((100.0 * h) / nlabels)
    
    print("Percentage of correct classifications: %4.1f %% out of %d images\n" % (p, nlabels))
    
run()