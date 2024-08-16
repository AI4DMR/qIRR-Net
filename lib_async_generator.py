# 2024/08/16
# public

import asyncio


async def async_generator(sync_generator):
    async def fun():
        try:
            return next(sync_generator)
        except StopIteration:
            raise StopAsyncIteration()

    task = asyncio.create_task(fun())
    while True:
        try:
            ret = await task
        except StopAsyncIteration:
            return
        else:
            task = asyncio.create_task(fun())
            yield ret
