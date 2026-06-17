"""
Bloomberg API connector via blpapi.
Supports BDP, BDS and BDH requests.
"""
import pandas as pd

# Tentative d'import blpapi
try:
    import blpapi
    BLPAPI_AVAILABLE = True
    SECURITY_DATA = blpapi.Name("securityData")
    SECURITY = blpapi.Name("security")
    FIELD_DATA = blpapi.Name("fieldData")
    OVERRIDES = blpapi.Name("overrides")
    FIELD_ID = blpapi.Name("fieldId")
    VALUE = blpapi.Name("value")
    DATE = blpapi.Name("date")
    ERROR_INFO = blpapi.Name("errorInfo")
    FIELD_EXCEPTIONS = blpapi.Name("fieldExceptions")
except ImportError:
    BLPAPI_AVAILABLE = False
    print("blpapi non disponible - mode offline uniquement")


class BloombergConnector:

    def __init__(self):
        self.connected = False
        self.session = None
        self.refDataSvc = None

    def connect(self):
        """Ouvre la session Bloomberg et le service refdata."""
        if not BLPAPI_AVAILABLE:
            print("blpapi non installe")
            self.connected = False
            return
        try:
            self.session = blpapi.Session()

            if not self.session.start():
                print("Impossible de demarrer la session Bloomberg")
                self.connected = False
                return

            if not self.session.openService("//blp/refdata"):
                print("Impossible d'ouvrir //blp/refdata")
                self.connected = False
                return

            self.refDataSvc = self.session.getService("//blp/refdata")
            self.connected = True
            print("Connecte a Bloomberg API - Session ouverte")

        except Exception as e:
            print(f"Erreur connexion Bloomberg: {e}")
            self.connected = False

    # ==================== BDP ====================
    def bdp(self, strSecurity, strFields, strOverrideField='', strOverrideValue=''):
        if not self.connected:
            print("Bloomberg non connecte")
            return {}

        request = self.refDataSvc.createRequest('ReferenceDataRequest')

        if isinstance(strFields, str):
            strFields = [strFields]
        if isinstance(strSecurity, str):
            strSecurity = [strSecurity]

        for strD in strFields:
            request.append('fields', strD)
        for strS in strSecurity:
            request.append('securities', strS)

        if strOverrideField != '':
            o = request.getElement('overrides').appendElement()
            o.setElement('fieldId', strOverrideField)
            o.setElement('value', strOverrideValue)

        self.session.sendRequest(request)

        list_msg = []
        data_dict = {}

        while True:
            event = self.session.nextEvent()
            if (event.eventType() != blpapi.event.Event.RESPONSE) & \
               (event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                continue
            for msg in blpapi.event.MessageIterator(event):
                list_msg.append(msg)
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

        for field in strFields:
            data_dict[field] = {}

        for msg in list_msg:
            security_data_arr = msg.getElement(SECURITY_DATA)
            for i in range(security_data_arr.numValues()):
                security_data = security_data_arr.getValueAsElement(i)
                ticker = security_data.getElementAsString(SECURITY)
                field_data = security_data.getElement(FIELD_DATA)
                for j in range(field_data.numElements()):
                    field_name = str(field_data.getElement(j).name())
                    field_value = field_data.getElement(j).getValue()
                    if field_name not in data_dict:
                        data_dict[field_name] = {}
                    data_dict[field_name][ticker] = field_value

        return data_dict

    # ==================== BDS ====================
    def bds(self, strSecurity, strFields, overrides=None):
        if not self.connected:
            print("Bloomberg non connecte")
            return pd.DataFrame()

        request = self.refDataSvc.createRequest('ReferenceDataRequest')

        if isinstance(strSecurity, str):
            strSecurity = [strSecurity]
        if isinstance(strFields, str):
            strFields = [strFields]

        for strF in strFields:
            request.append('fields', strF)
        for strS in strSecurity:
            request.append('securities', strS)

        if overrides:
            overridesElement = request.getElement(OVERRIDES)
            for fieldId, value in overrides.items():
                override = overridesElement.appendElement()
                override.setElement(FIELD_ID, fieldId)
                override.setElement(VALUE, value)

        self.session.sendRequest(request)

        list_msg = []
        while True:
            event = self.session.nextEvent()
            if (event.eventType() != blpapi.event.Event.RESPONSE) & \
               (event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                continue
            for msg in blpapi.event.MessageIterator(event):
                list_msg.append(msg)
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

        data_list = []
        for msg in list_msg:
            security_data_array = msg.getElement(SECURITY_DATA)
            for i in range(security_data_array.numValues()):
                security_data_item = security_data_array.getValueAsElement(i)
                field_data = security_data_item.getElement(FIELD_DATA)
                for j in range(field_data.numElements()):
                    field_element = field_data.getElement(j)
                    for k in range(field_element.numValues()):
                        row_element = field_element.getValueAsElement(k)
                        row_dict = {}
                        for m in range(row_element.numElements()):
                            col_elem = row_element.getElement(m)
                            row_dict[str(col_elem.name())] = col_elem.getValue()
                        data_list.append(row_dict)

        df = pd.DataFrame(data_list) if data_list else pd.DataFrame()
        return df

    # ==================== BDH ====================
    def bdh(self, strSecurity, strFields, startdate, enddate, per='DAILY',
            perAdj='CALENDAR', days='NON_TRADING_WEEKDAYS', fill='PREVIOUS_VALUE', curr=None,
            adjust=True):
        if not self.connected:
            print("Bloomberg non connecte")
            return {}

        request = self.refDataSvc.createRequest('HistoricalDataRequest')

        if isinstance(strFields, str):
            strFields = [strFields]
        if isinstance(strSecurity, str):
            strSecurity = [strSecurity]

        for strF in strFields:
            request.append('fields', strF)
        for strS in strSecurity:
            request.append('securities', strS)

        request.set('startDate', startdate.strftime('%Y%m%d'))
        request.set('endDate', enddate.strftime('%Y%m%d'))
        request.set('periodicitySelection', per)
        request.set('periodicityAdjustment', perAdj)
        request.set('nonTradingDayFillMethod', fill)
        request.set('nonTradingDayFillOption', days)
        # adjust=True -> total-return-adjusted prices (dividends + splits).
        # adjust=False -> raw closing prices with no corporate-action adjustment.
        request.set('adjustmentNormal', adjust)
        request.set('adjustmentAbnormal', adjust)
        request.set('adjustmentSplit', adjust)
        request.set('adjustmentFollowDPDF', False)
        if curr is not None:
            request.set('currency', curr)

        self.session.sendRequest(request)

        list_msg = []
        data_dict = {}

        while True:
            event = self.session.nextEvent()
            if (event.eventType() != blpapi.event.Event.RESPONSE) & \
               (event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                continue
            for msg in blpapi.event.MessageIterator(event):
                list_msg.append(msg)
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

        for msg in list_msg:
            security_data = msg.getElement(SECURITY_DATA)
            ticker = security_data.getElement(SECURITY).getValue()
            field_data = security_data.getElement(FIELD_DATA)

            for field_ele in field_data:
                dat_date = field_ele.getElement(0).getValue()
                for i in range(1, field_ele.numElements()):
                    field_name = str(field_ele.getElement(i).name())
                    field_value = field_ele.getElement(i).getValue()
                    if field_name not in data_dict:
                        data_dict[field_name] = {}
                    if ticker not in data_dict[field_name]:
                        data_dict[field_name][ticker] = {}
                    data_dict[field_name][ticker][dat_date] = field_value

        return data_dict

    def disconnect(self):
        if self.session:
            self.session.stop()
            print("Session Bloomberg fermee")
        self.connected = False
