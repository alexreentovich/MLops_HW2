'''
The code below is taken from:
https://github.com/python-restx/flask-restx/issues/115#issuecomment-630073568
'''
from flask_restx import fields


class DictItem(fields.Raw):
    """Marshal a value as a dict"""

    def output(self, key, obj, *args, **kwargs):
        try:
            dct = getattr(obj, self.attribute)
        except AttributeError:
            return {}
        return dct or {}
