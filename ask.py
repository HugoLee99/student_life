pd = [{'static': None, 'unlabeled': 12},{'static': 23, 'unlabeled': 12},{'static': 45, 'unlabeled': None}]
result = (
        [data['static'] for data in pd if data['static'] is not None],
        [data['unlabeled'] for data in pd if data['unlabeled'] is not None],
    )
print(result)