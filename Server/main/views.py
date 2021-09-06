import json

from django.shortcuts import render
from main.models import ReListing


def splash(request):
    return render(request, "splash.html", {})


def post(request, id):
    listing = ReListing.objects.get(id=id)

    s = ""
    if listing.luxury:
        s = s + "Luxury "
    if listing.listedbuilding:
        s = s + "Listed "
    if listing.neoclassico:
        s = s + "Neo-classical "
    if listing.newlybuilt:
        s = s + "New-built "
    if listing.unfinished:
        s = s + "Under construction "
    if listing.summerhouse:
        s = s + "Summer house "
    if 0 < float(listing.distancefromsea) < 2001:
        s = s + "(%s m from sea) " % listing.distancefromsea
    if listing.orientation:
        s = s + "with %s orientation, " % listing.orientation
    if listing.views:
        if not listing.orientation:
            s = s + "with "
        s = s + "great views, "
    if listing.corner:
        s = s + "on corner, "
    if listing.roadfront:
        s = s + "on main road, "
    if listing.rooftop or listing.swimmingpool or listing.storage:
        s = s + "Facilities include: "
        if listing.rooftop:
            s = s + "roof top, "
        if listing.swimmingpool:
            s = s + "swimming pool, "
        if listing.storage:
            s = s + "storage space, "
    if listing.renovated:
        s = s + "Last renovated: %s, " % listing.yearrenovated
    elif listing.needsrenovation:
        if listing.unfinished:
            pass
        else:
            s = s + "Requires renovation, "
    if listing.studentaccomodation or listing.professionaluse:
        s = s + "Alternative use: "
        if listing.studentaccomodation:
            s = s + "student accommodation, "
        if listing.professionaluse:
            s = s + "business use, "
    if listing.energycat:
        s = s + "Energy category: %s " % listing.energycat

    latlon = str(listing.lat) + "," + str(listing.lng)

    geoloc = "https://www.google.com/maps/embed/v1/place?key=AIzaSyAiqM6Yh8fv2xOCNh9Av_mfF5fUx7FE0LA&q=" + latlon
    listing.pricepersqm_p = listing.price_p / listing.area  # show implied price per sqm rather than one predicted by model
    context = {
        'listing': listing,
        'features': s,
        'geoloc': geoloc,
    }
    return render(request, "post.html", context)


def search(request):
    home_type = request.POST.get('home_type')
    no_listings = request.POST.get('no_listings')
    std_min = request.POST.get('std_min')
    std_max = request.POST.get('std_max')

    if std_max == "":
        std_max = 20
    if std_min == "":
        std_min = -20
    if no_listings == "":
        no_listings = 100

    if home_type == "":
        listings = ReListing.objects.order_by('price_stderr').filter(price_stderr__lte=float(std_max),
                                                                     price_stderr__gte=float(std_min))[:int(no_listings)]
    else:
        listings = ReListing.objects.order_by('price_stderr').filter(hometype=home_type,
                                                                     price_stderr__lte=float(std_max),
                                                                     price_stderr__gte=float(std_min))[:int(no_listings)]

    list_pins = []
    for listing in listings:
        list_pins.append(
            {'listingID': str(listing.id), 'listingName': str(listing), 'location': (listing.lat, listing.lng),
             'price': int(listing.price), 'pricepersqm': int(listing.pricepersqm),
             'price_p': int(listing.price_p), 'pricepersqm_p': int(listing.price_p / listing.area)}
        )
    info_box_template = """
        <dl>
        <dt><style="font-size:8">Listing: </dt><dd><a href="post/{listingID}">{listingName}</a></dd>
        <dt>Price: </dt><dd>{price}</dd>
        <dt> Price / sqm: </dt><dd>{pricepersqm}</dd>
        <dt>Model price: </dt><dd>{price_p}</dd>
        <dt> Model price / sqm: </dt><dd>{pricepersqm_p}</dd>
        </dl>
        """
    pin_locations = [pin['location'] for pin in list_pins]
    pin_info = [info_box_template.format(**pin) for pin in list_pins]
    pin_ids = ["post/" + str(l.id) for l in listings]
    return render(request, "search.html", {'listings': listings,
                                           'locations': json.dumps(pin_locations),
                                           'pin_info': json.dumps(pin_info),
                                           'pin_ids': pin_ids,
                                           })
