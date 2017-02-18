using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace Retia.Gui.Messages
{
    public class Post
    {
        private readonly ConcurrentDictionary<string, List<Action<object>>> _subscribers = new ConcurrentDictionary<string, List<Action<object>>>();

        public static Post Box { get; } = new Post();

        public void Subscribe<T>(string message, Action<T> handler)
        {
            Subscribe(message, (object arg) => handler((T)arg));
        }

        public void Subscribe(string message, Action<object> handler)
        {
            var list = _subscribers.GetOrAdd(message, msg => new List<Action<object>>());
            list.Add(handler);
        }

        public void Publish(string message, object arg)
        {
            List<Action<object>> list;
            if (_subscribers.TryGetValue(message, out list))
            {
                for (int i = 0; i < list.Count; i++)
                {
                    list[i](arg);
                }
            }
        }
    }
}