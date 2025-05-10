#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from dotenv import load_dotenv


# load_dotenv()
load_dotenv()
""" Bot Configuration """


class DefaultConfig:
    """ Bot Configuration """
    azure_app_id=os.getenv("AZURE_APP_ID")
    azure_app_password=os.getenv("AZURE_APP_PASSWORD")
    azure_tenant_id=os.getenv("AZURE_TENANT_ID")
    
    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", azure_app_id)
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", azure_app_password)
    APP_TYPE = os.environ.get("MicrosoftAppType", "MultiTenant")
    APP_TENANTID = os.environ.get("MicrosoftAppTenantId", azure_tenant_id)