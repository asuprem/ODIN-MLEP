

def generate_cell(N, E, coef=0.04166666666667):
    from decimal import Decimal

    if coef<0.04166666666667: coef = 0.04166666666667
    if coef>1: coef = 1
    try:
        row = int(round(Decimal(Decimal(90.0)+Decimal(N))/Decimal(coef)))
#         row = int(round(Decimal(Decimal(85.0)-Decimal(N))/Decimal(coef)))
    except Exception, e:
        print e
    if row<0:
        print 'Hello, World!'
    col = int(round(Decimal(Decimal(180.0)+Decimal(E))/Decimal(coef)))
    key = str(row)+'_'+str(col)

    return key

def lookup_address_only(address, API_KEY):
    import httplib, urllib
    
    host = 'maps.googleapis.com'
    params = {'address': address, 'key': API_KEY}
    url = '/maps/api/geocode/json?'+urllib.urlencode(params)
    req = httplib.HTTPSConnection(host)
    req.putrequest('GET', url)
    req.putheader('Host', host)
    req.endheaders()
    resp = req.getresponse()
    if resp.status==200:
        result = json.load(resp, encoding='UTF-8')
        if 'results' in result:
            results = result['results']
            if len(results) > 0:
                item = results[0]
                if 'geometry' in item:
                    geometry = item['geometry']
                    if 'location' in geometry:
                        location = geometry['location']
                        lat = location['lat']
                        lng = location['lng']
            else:
                return None, None
    else:
        return None, None
    return lat, lng