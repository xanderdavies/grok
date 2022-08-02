def add_tags(*args_and_tags):
    tags = []
    name = ""
    for tag, condition in args_and_tags:
        if condition:
            tags.append(tag)
            name += tag
    return name, tags