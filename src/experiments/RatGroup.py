ALL_RATS = "ALL_RATS"
ALL_REGIONS = "ALL_REGIONS"
class RatGroup:
    def __init__(self, rats=[], regions=[], group_name = None):
        if rats == []: raise ValueError("Must specify rats")
        if regions == []: raise ValueError("Must specify regions")
        self.rats = rats
        self.regions = regions

        self.group_name = group_name
    
    def include_rat(self, rat_id):
        if self.rats == ALL_RATS: return True
        return rat_id in self.rats
    
    def include_region(self, region_name):
        if self.regions == ALL_REGIONS: return True
        return region_name in self.regions


## Some premade RatGroups for testing
contralesional_group = RatGroup(rats=ALL_RATS, 
                                regions=["contra_inner", "contra_outer"],
                                group_name = "Contralesional"
                                )

ipsilesional_group = RatGroup(rats=ALL_RATS, 
                                regions=["ipsi_inner", "ipsi_outer"],
                                group_name = "Ipsilesional"
                                )
