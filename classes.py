class ListDict(list):
    """ This subclass of list inherits list like slicing and adds support for
        dictionary style data access by
        overwriting the __getitem__ method:
            list like slicing:  Clients(9)[7],       Clients(9)[3:6]
            dict style access:  Clients(9)['Alice'], Clients(9)[['Dave', 'Carol']]
    """
    # children of a listdict must contain a key attribute for dict style access
    def keys(self):
        return [child.key for child in self]

    def __getitem__(self, query):
        # handle integer input
        if isinstance(query, int):
            return list(self)[query]
        # handle slicing input
        if isinstance(query, slice):
            return list(self)[query]
        # handle single string input
        if isinstance(query, str):
            if query in [child.key for child in self]:
                return [child for child in self if child.key == query][0]
            else:
                raise KeyError(query)
        # handle list of strings input
        if isinstance(query, list):
            r = [self[q] for q in query if q in self.keys()]
            if not r:
                raise KeyError(query)
            elif len(r) < len(query):
                print('KeyWarning:', [q for q in query if q not in self.keys()])
            return r


class Cohorts(ListDict):
    """ You can import `cohorts` and `colors` from the classes file.
    These objects are used to enable central configuration of properties of these abstractions
     (*e.g. one designated color or long name per site throughout the project*)."""
    def __init__(self, *args):
        self.extend(args)
        attrs = [[atr for atr in dir(i) if not atr[0] == '_'] for i in self]
        self.attrs = {item for sublist in attrs for item in sublist}
        self.format = lambda val: f'{val:.0%}'

    def iter(self, *args):
        # Iterate over attribute
        if len(args) == 1:
            return [getattr(i, args[0]) for i in self[:]]
        else:
            return zip(*[self.iter(arg) for arg in args])

    def get(self, field):
        # Get child by attribute
        if isinstance(field, list):
            return Cohorts(*[self.get(f) for f in field])
        for mod in self:
            for attr in self.attrs:
                if hasattr(mod, attr):
                    if getattr(mod, attr) == field:
                        return mod


class Cohort:
    def __init__(self, key: str, idx: int, name: str, color: str, design='Naturalistic', mu=None, sd=None):
        self.key = key
        self.idx = idx
        self.name = name
        self.color = color
        self.design = design
        self.treatment_duration_mu = mu
        self.treatment_duration_sd = sd


class Colors:
    def __init__(self):
        self.dark = '#231f20'
        self.tan = '#f3f3e9'
        self.gray = '#818181'
        self.green = '#009e73'
        self.yellow = '#f0e442'
        self.blue = '#0072b2'
        self.lavender = '#cc79a7'
        self.orange = '#d55e00'
        self.teal = '#36989a'


colors = Colors()

cohorts = Cohorts(
    Cohort(key='AFFDIS',      idx=0, name='AFFDIS', color=colors.green,             mu=5.1, sd=0.7),
    Cohort(key='CARDIFF',     idx=1, name='Cardiff', color=colors.dark),
    Cohort(key='MOODS',       idx=2, name='DEP-ARREST-CLIN', color=colors.orange,   mu=12., sd=0.0),
    Cohort(key='Hiroshima',   idx=3, name='Hiroshima cohort', color=colors.yellow,  mu=6., sd=0.0, design='RCT'),
    Cohort(key='Melb',        idx=4, name='Melbourne', color=colors.blue,           mu=12., sd=0.0, design='RCT'),
    Cohort(key='Minnesota',   idx=5, name='Minnesota', color=colors.teal,           mu=9.9, sd=2.0),
    Cohort(key='SanRaffaele', idx=6, name='Milano OSR', color=colors.lavender,      mu=4.2, sd=0.7),
    Cohort(key='TIGER',       idx=7, name='Stanford TIGER', color=colors.dark),
    Cohort(key='SF',          idx=8, name='UCSF Adolescent MDD', color=colors.dark),
)
