def has_context(x):
    '''
    input
        x: takes in the name of the out file
    output
        returns 1 if "cddc" or "cdstc7" is present in the name of the out file
        else returns 0
    '''
    x = x.lower()
    if("cddc" in x or "cdstc7" in x):
        return 1
    return 0


if __name__ == "__main__":
    print(has_context("out.seen.ddc.mpc"))
    print(has_context("out.seen.cddc.gpt2.sloss"))